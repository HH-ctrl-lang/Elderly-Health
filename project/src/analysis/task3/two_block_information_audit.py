"""任务3 第11步：冻结双块模型，区分读出不足、信息分配与语义信息保留问题。

复用任务5第3步的无MMD/条件MMD模型，不重新训练编码器，不继续教育局部调参。
这是失败机制诊断，不是新论文复现，也不将任何表示块认定为因果变量。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sparse_invariance_feature_pilot as base
from elderly_health.data import prepare_splits
from elderly_health import conditional_mmd_adapter as adapter

ARMS = {'two_block_control':0., 'two_block_conditional_mmd':.1}
SPACES = ('raw19','prediction8','residual8','joint16','raw_joint35')
SEMANTIC_SPACES = ('observed_categories','raw19','prediction8','residual8','joint16')
REGIONS = ('east','central','west')
RIDGE_ALPHA = 1.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifact(path):
    """任务3 第11步：拒绝缺失文件或Git LFS指针，不静默重新训练历史模型。"""
    if not path.is_file():
        raise FileNotFoundError(f'缺少历史文件：{path}。请保留任务5第3步完整结果。')
    with path.open('rb') as stream:
        pointer = stream.read(100).startswith(b'version https://git-lfs.github.com/spec/')
    if pointer:
        raise ValueError('历史文件仍是LFS指针，请先运行：git lfs pull --include="outputs/analysis_2611/task5_step3_health_disentanglement_screen/**"')


def state_digest(model):
    """任务3 第11步：拟合新读出器前后校验编码器参数完全不变。"""
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode()); value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def load_model(path, expected_weight):
    """任务3 第11步：仅加载张量与基础类型；严格核对历史结构与约束强度。"""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    settings = payload['settings']
    for name, expected in (('prediction_block_dim',8),('residual_block_dim',8),('positive_weight',3.)):
        if settings.get(name) != expected:
            raise ValueError(f'历史模型设置不匹配：{name}')
    if payload.get('mmd_weight') != expected_weight:
        raise ValueError('历史模型的MMD强度不匹配')
    model = adapter.build_model(len(base.FEATURES))
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def extract(model, x):
    """任务3 第11步：沿用历史256行推断批次，提取表示和原分类头概率。"""
    import torch
    prediction, residual, original = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),256):
            values = torch.as_tensor(x[start:start+256])
            a = model['prediction_encoder'](values)
            b = model['residual_encoder'](values)
            prediction.append(a.numpy()); residual.append(b.numpy())
            original.append(torch.sigmoid(model['classifier'](a)).numpy().ravel())
    a = np.concatenate(prediction); b = np.concatenate(residual)
    joint = np.column_stack([a,b])
    return dict(raw19=x,prediction8=a,residual8=b,joint16=joint,
                raw_joint35=np.column_stack([x,joint])), np.concatenate(original)


def check_splits(folder, frames, splits, preprocessor):
    """任务3 第11步：逐折核对人员顺序哈希、人数、标签数和来源预处理参数。"""
    manifest = json.loads((folder/'split_manifest.json').read_text(encoding='utf-8'))
    ids = [set(s.sample_ids) for s in splits]
    if any(a & b for i,a in enumerate(ids) for b in ids[i+1:]):
        raise RuntimeError('来源训练、验证与目标人群发生重叠')
    for name, part in zip(('train','validation','test'),splits):
        actual = dict(n=len(part.y),positive_count=int(part.y.sum()),
            sample_ids_sha256=hashlib.sha256(np.asarray(part.sample_ids,dtype=np.int64).tobytes()).hexdigest())
        if actual != manifest[name]: raise RuntimeError(f'历史划分不匹配：{name}')
    with np.load(folder/'preprocessing.npz',allow_pickle=False) as old:
        values = dict(median=preprocessor.imputer.statistics_,mean=preprocessor.scaler.mean_,scale=preprocessor.scaler.scale_)
        if any(not np.allclose(old[k],v,atol=1e-9,rtol=1e-9) for k,v in values.items()):
            raise RuntimeError('历史预处理与当前来源拟合结果不匹配')


def evaluate(context, stage, frame, p):
    """任务3 第11步：合并、站点和手机子组分别记录，宏平均另算。"""
    return [dict(**context,stage=stage,group=group,subgroup=subgroup,
                 **base.aggregate_metrics(frame.label.to_numpy(dtype=int)[keep],p[keep]))
            for group,subgroup,keep in base.cohort_masks(frame)]


def representation_stats(context, stage, space, x):
    """任务3 第11步：有效秩和方差只作数值诊断，不作为因果崩溃的单独判据。"""
    centered = np.asarray(x,dtype=float)-np.mean(x,axis=0,dtype=float)
    singular = np.linalg.svd(centered,compute_uv=False)
    power = singular**2
    total = float(power.sum())
    shares = power/total if total>1e-20 else np.zeros_like(power)
    positive = shares[shares>0]
    entropy_rank = float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.
    tolerance = (singular[0] if len(singular) else 0.)*max(x.shape)*np.finfo(np.float32).eps
    return dict(**context,stage=stage,space=space,columns=x.shape[1],
        numeric_rank_at_float32_precision=int((singular>tolerance).sum()),
        effective_rank=entropy_rank,top_component_fraction=float(shares.max()) if len(shares) else 0.,
        mean_std=float(x.std(0).mean()),near_constant_columns=int((x.std(0)<1e-6).sum()),
        prediction_layernorm_constraint=(space in ('prediction8','joint16','raw_joint35')))


def fit_semantic_readers(frame, x):
    """任务3 第11步：只在来源训练中拟合原变量类别读出器；缺失值不充当真值。"""
    readers = {}
    for feature in base.FEATURES:
        truth = frame[feature].to_numpy(dtype=float); keep = np.isfinite(truth)
        levels, counts = np.unique(truth[keep],return_counts=True)
        if not len(levels): raise ValueError('来源特征整列缺失')
        labels = np.searchsorted(levels,truth[keep])
        reader = None
        if len(levels)>1:
            reader = RidgeClassifier(alpha=RIDGE_ALPHA,class_weight='balanced').fit(x[keep],labels)
        readers[feature] = (levels,reader,float(levels[counts.argmax()]))
    return readers


def balanced_accuracy(y, p):
    """任务3 第11步：逐真实类别平均召回，包括来源未见类别，避免静默删难样本。"""
    return float(np.mean([np.mean(p[y==level]==level) for level in np.unique(y)]))


def semantic_metrics(context, stage, frame, x, readers):
    """任务3 第11步：只评估类别可读性，不把类别编号距离解释为医学距离。"""
    rows = []
    masks = [('REGION_ALL',np.ones(len(frame),dtype=bool))]
    masks += [(str(s),frame[base.ENVIRONMENT].to_numpy()==s) for s in sorted(frame[base.ENVIRONMENT].unique())]
    for feature,(levels,reader,majority) in readers.items():
        pred = np.full(len(x),levels[0]) if reader is None else levels[reader.predict(x).astype(int)]
        truth = frame[feature].to_numpy(dtype=float)
        for group, keep in masks:
            keep = keep & np.isfinite(truth)
            if not keep.any():continue
            y,p = truth[keep],pred[keep]
            rows.append(dict(**context,stage=stage,group=group,feature=feature,observed_n=len(y),
                source_categories=len(levels),evaluation_categories=len(np.unique(y)),
                unseen_category_n=int((~np.isin(y,levels)).sum()),
                accuracy=float(np.mean(y==p)),balanced_accuracy=balanced_accuracy(y,p),
                majority_accuracy=float(np.mean(y==majority)),
                majority_balanced_accuracy=balanced_accuracy(y,np.full(len(y),majority))))
    return rows


def macro_table(metrics):
    part = metrics[(metrics.stage=='test') & (metrics.group!='REGION_ALL')]
    rows = []
    for scope in (*REGIONS,'ALL'):
        values = part if scope=='ALL' else part[part.held_out==scope]
        for (arm,head,subgroup),group in values.groupby(['arm','head','subgroup']):
            rows.append(dict(scope=scope,arm=arm,head=head,subgroup=subgroup,
                site_count=len(group),**group[['auc','auprc','brier']].mean().to_dict()))
    return pd.DataFrame(rows)


def reference_check(rows, historical):
    """任务3 第11步：原头必须重现历史站点指标，拒绝加载错误模型后继续解释。"""
    now = rows[(rows.stage=='test') & (rows['head']=='original_head') &
               (rows.subgroup=='all') & (rows.group!='REGION_ALL')]
    old = historical[historical.stage=='before'].rename(columns={'held_out_region':'held_out','model':'arm','site_code':'group'})
    joined = now.merge(old,on=['held_out','arm','group'],suffixes=('_now','_old'),validate='one_to_one')
    expected = len(REGIONS)*len(ARMS)*4
    if len(now)!=expected or len(joined)!=expected or len(old)!=expected:
        raise RuntimeError(f'历史站点结果未完整匹配：预期{expected}行，当前{len(now)}行，匹配{len(joined)}行')
    result = []
    for metric in ('auc','auprc','brier'):
        delta = np.abs(joined[metric+'_now']-joined[metric+'_old'])
        result.append(dict(metric=metric,compared_rows=len(joined),max_absolute_difference=float(delta.max()),
                           passed=bool(np.isfinite(delta).all() and (delta<1e-6).all())))
    return pd.DataFrame(result)


def draw(metrics,macro,semantic,diagnostics,output):
    """任务3 第11步：四图对应读出、块分配、语义保留和数值退化，全部地区展示。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    heads = ['original_head','prediction8','residual8','joint16','raw19','raw_joint35']
    fig,axes = plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,arm in zip(axes,ARMS):
        part = macro[(macro.arm==arm)&(macro.subgroup=='all')]
        values = part.pivot(index='scope',columns='head',values='auc').reindex(index=[*REGIONS,'ALL'],columns=heads)
        values.plot.bar(ax=ax,rot=0);ax.set_title(arm);ax.set_ylabel('Equal-site macro AUC');ax.set_ylim(0,1);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/01_original_and_probe_auc.png',dpi=180);plt.close(fig)
    site = metrics[(metrics.stage=='test')&(metrics.subgroup=='all')&(metrics.group!='REGION_ALL')]
    order = [(region,str(s)) for region in REGIONS for s in sorted(site.loc[site.held_out==region,'group'].unique())]
    deltas = []
    for arm in ARMS:
        pivot = site[site.arm==arm].pivot(index=['held_out','group'],columns='head',values='auc').reindex(pd.MultiIndex.from_tuples(order))
        deltas.append(np.column_stack([pivot.prediction8-pivot.original_head,pivot.joint16-pivot.prediction8,pivot.raw_joint35-pivot.joint16]))
    limit=max(.01,max(float(np.nanmax(np.abs(a))) for a in deltas))
    fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,arm,values in zip(axes,ARMS,deltas):
        im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(3));ax.set_xticklabels(['Refit pred - original','Joint - prediction','Raw+joint - joint'],rotation=15,ha='right',fontsize=8)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([f'{r}: {s}' for r,s in order]);ax.set_title(arm)
        for i in range(len(order)):
            for j in range(3):
                value=values[i,j];ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.65*limit else 'black')
    fig.subplots_adjust(left=.16,right=.86,wspace=.1,bottom=.13,top=.94)
    cax=fig.add_axes([.9,.2,.015,.6]);fig.colorbar(im,cax=cax,label='Site AUC difference')
    fig.savefig(output/'figures/02_site_information_access_gaps.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,arm in zip(axes,ARMS):
        part=semantic[(semantic.arm==arm)&(semantic.stage=='test')&(semantic.group!='REGION_ALL')]
        values=part.groupby(['feature','space']).balanced_accuracy.mean().unstack().reindex(index=base.FEATURES,columns=SEMANTIC_SPACES)
        im=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(len(SEMANTIC_SPACES)));ax.set_xticklabels(SEMANTIC_SPACES,rotation=25,ha='right',fontsize=8)
        ax.set_yticks(range(len(base.FEATURES)));ax.set_yticklabels(base.FEATURES,fontsize=9);ax.set_title(arm)
    fig.subplots_adjust(left=.17,right=.86,wspace=.1,bottom=.17,top=.94)
    cax=fig.add_axes([.9,.2,.015,.6]);fig.colorbar(im,cax=cax,label='Equal-site mean balanced accuracy: reading original categories')
    fig.savefig(output/'figures/03_semantic_feature_readability.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,space in zip(axes,('prediction8','residual8','joint16')):
        part=diagnostics[(diagnostics.stage=='test')&(diagnostics.space==space)]
        values=part.pivot(index='held_out',columns='arm',values='effective_rank').reindex(REGIONS)
        values.plot.bar(ax=ax,rot=0);ax.set_title(space);ax.set_ylabel('Entropy effective rank');ax.legend(fontsize=6)
    fig.suptitle('Rank is a numerical diagnostic; prediction LayerNorm creates a coordinate dependence')
    fig.tight_layout();fig.savefig(output/'figures/04_representation_rank_checks.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--history',type=Path,default=base.ROOT/'outputs/analysis_2611/task5_step3_health_disentanglement_screen')
    parser.add_argument('--output',type=Path,default=base.ROOT/'outputs/analysis_2611/task3_step11_two_block_information_audit')
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args()
    if args.steps<500:raise ValueError('读出器至少优化500轮')
    if digest(args.data)!=base.EXPECTED_HASH:raise ValueError('固定数据版本不一致')
    required=[args.history/'experiment_design.json',args.history/'site_metrics_before_after.csv',args.history/'completion.json']
    required += [args.history/region/name for region in REGIONS for name in
                 ('preprocessing.npz','split_manifest.json',*(a+'.pt' for a in ARMS))]
    for path in required:check_artifact(path)
    design=json.loads((args.history/'experiment_design.json').read_text(encoding='utf-8'))
    if design['data_sha256']!=base.EXPECTED_HASH or not json.loads((args.history/'completion.json').read_text())['completed']:
        raise ValueError('历史实验未完成或使用了不同的数据')
    import torch
    torch.set_num_threads(2)
    frame=base.attach_regions(base.read_raw(args.data))
    historical=pd.read_csv(args.history/'site_metrics_before_after.csv',dtype={'site_code':str})
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    metadata=dict(completed=False,encoder_retrained=False,split_seed=42,probe_initialization_seed=42,
        prediction_probe_steps=args.steps,prediction_probe_positive_weight=base.POSITIVE_WEIGHT,
        prediction_probe_ridge=base.RIDGE,prediction_probe_environment_weighting='equal_source_site',
        prediction_probe_selection='source_validation_equal_site_macro_auc_then_weighted_bce',
        semantic_probe='class-balanced RidgeClassifier; fixed alpha 1; observed training categories only',
        arms=ARMS,spaces=SPACES,semantic_spaces=SEMANTIC_SPACES,
        target_used_for_training_or_selection=False,individual_records_exported=False,
        input_sha256=base.EXPECTED_HASH,artifact_sha256={str(p.relative_to(args.history)):digest(p) for p in required},
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        code_sha256={Path(m.__file__).name:digest(Path(m.__file__)) for m in (base,adapter)},
        runner_sha256=digest(Path(__file__)),
        research_links=['https://proceedings.mlr.press/v267/gamella25a.html',
            'https://proceedings.iclr.cc/paper_files/paper/2025/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html',
            'https://proceedings.iclr.cc/paper_files/paper/2025/hash/38fc6254f73450813db3b3e04397a9fc-Abstract-Conference.html'],
        limitations=['本步是诊断而非论文方法复现；不把可解释性或预测提升等同因果识别，原变量可读性也不是论文的神经元单义性指标。',
            '单个历史编码器种子，来源验证已用于编码器选模，再用于读出选模；目标已反复探索。',
            '原头与重拟合头的损失环境权重、正则和选模政策不同，改善只能定位到读出及其训练政策这一层。',
            '表示维数与读出容量不同，拼接收益不能单独证明编码器丢失了信息。线性类别不可读不等于信息彻底不存在。',
            '原类别正对照通过直接读取输入检验探针；它不是一个疾病预测算法。缺失原特征不当作恢复真值。',
            '低维有效秩并非表征崩溃，预测块LayerNorm本身限制其坐标自由度；残余块没有因果或环境专属保证。',
            '无MMD与MMD模型各自采用历史选出的轮次；本审计不能把全部差异单独归因MMD。'])
    base.dump(args.output/'experiment_metadata.json',metadata)
    metrics,semantic,diagnostics,selections,controls=[],[],[],[],[]
    for region in REGIONS:
        frames=base.held_out_split(frame,region,seed=42,column=base.REGION)
        splits,preprocessor=prepare_splits(frames,environment_column=base.REGION)
        check_splits(args.history/region,frames,splits,preprocessor)
        category_basis,train_categories=base.fit_basis(frames[0],'categorical_grouped')
        categories=[train_categories[:,:-1]]+[base.transform_basis(f,category_basis)[0][:,:-1] for f in frames[1:]]
        for arm,weight in ARMS.items():
            print(f'任务3 第11步：留出{region}，冻结模型={arm}',flush=True)
            model=load_model(args.history/region/(arm+'.pt'),weight);before=state_digest(model)
            extracted=[extract(model,s.x) for s in splits]
            raw_spaces=[r[0] for r in extracted]
            predictions={'original_head':[r[1] for r in extracted]}
            standardized={};parameters={}
            for space in (*SPACES,'observed_categories'):
                arrays=categories if space=='observed_categories' else [s[space] for s in raw_spaces]
                scaler=StandardScaler().fit(arrays[0])
                standardized[space]=[scaler.transform(x).astype(float) for x in arrays]
                if space=='observed_categories':continue
                context=dict(held_out=region,arm=arm,head=space)
                values=standardized[space];x=np.column_stack([values[0],np.ones(len(values[0]))])
                xv=np.column_stack([values[1],np.ones(len(values[1]))])
                result=base.train_candidate(x,splits[0].y.astype(int),frames[0][base.ENVIRONMENT].to_numpy(),
                    xv,splits[1].y.astype(int),frames[1][base.ENVIRONMENT].to_numpy(),
                    [np.array([i]) for i in range(x.shape[1]-1)],42,0.,None,args.steps)
                predictions[space]=[expit(np.column_stack([a,np.ones(len(a))])@result['weights']) for a in values]
                parameters[space]=(scaler,result)
                selections.append(dict(**context,epoch=result['epoch'],source_macro_auc=result['source_macro'],source_weighted_bce=result['source_loss']))
            folder=args.output/region/arm;folder.mkdir(parents=True)
            for space,(scaler,result) in parameters.items():
                np.savez(folder/(space+'_readout.npz'),weights=result['weights'],mean=scaler.mean_,scale=scaler.scale_)
                base.dump(folder/(space+'_selection.json'),{k:v for k,v in result.items() if k!='weights'})
            # 任务3 第11步：所有疾病读出器来源选模结束后才评估目标标签。
            for head,probabilities in predictions.items():
                context=dict(held_out=region,arm=arm,head=head)
                for stage,part,p in zip(('source_train','source_validation','test'),frames,probabilities):
                    metrics.extend(evaluate(context,stage,part,p))
            for space in ('prediction8','residual8','joint16'):
                for stage,values in zip(('source_train','source_validation','test'),raw_spaces):
                    diagnostics.append(representation_stats(dict(held_out=region,arm=arm),stage,space,values[space]))
            for space in SEMANTIC_SPACES:
                readers=fit_semantic_readers(frames[0],standardized[space][0])
                for stage,part,values in zip(('source_validation','test'),frames[1:],standardized[space][1:]):
                    semantic.extend(semantic_metrics(dict(held_out=region,arm=arm,space=space),stage,part,values,readers))
            after=state_digest(model)
            if before!=after:raise RuntimeError('冻结编码器参数被修改')
            controls.append(dict(held_out=region,arm=arm,encoder_unchanged=True,state_sha256=before,split_verified=True))
            for name,values in dict(metrics=metrics,semantic_probe_metrics=semantic,
                representation_diagnostics=diagnostics,source_only_selection=selections,control_checks=controls).items():
                pd.DataFrame(values).to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    table=pd.DataFrame(metrics);macro=macro_table(table)
    checks=reference_check(table,historical)
    checks.to_csv(args.output/'historical_head_reproduction.csv',index=False,encoding='utf-8-sig')
    if not checks.passed.all():raise RuntimeError('历史原头未复现，已保存核查表，请暂停解释其他读出结果')
    macro.to_csv(args.output/'region_and_global_macro.csv',index=False,encoding='utf-8-sig')
    sem=pd.DataFrame(semantic)
    sem[(sem.stage=='test')&(sem.group!='REGION_ALL')].groupby(['arm','space','feature'],as_index=False)[
        ['balanced_accuracy','accuracy','majority_balanced_accuracy']].mean().to_csv(args.output/'semantic_feature_macro.csv',index=False,encoding='utf-8-sig')
    draw(table,macro,sem,pd.DataFrame(diagnostics),args.output)
    metadata.update(completed=True,frozen_encoder_count=len(controls),new_outcome_readout_count=len(selections))
    base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第11步完成：冻结6个历史模型、拟合30个疾病读出器，输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
