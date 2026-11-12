"""任务3 第16步：复用完整V5检查点，审计损失梯度的实际作用路径。

已核对实验2610、2611和门控/特征置换、表示探针、损失日志任务。
本步只新增逐损失、逐参数组的梯度范数、分类方向关系与门控更新方向；
不重复训练、置换重要性、表征秩或算法比较，不选择新检查点。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT.parent
sys.path.insert(0,str(PROJECT/'src'))
from elderly_health import losses,models
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES

REGIONS=('east','central','west')
MODES=('eval','train_dropout')
TERMS=('main_prediction','zc_prediction','sparse','irm','environment','independence')
REGULARIZERS=('sparse','irm','environment','independence','all_regularizers')
GROUPS=('gate','encoder_c','encoder_s','predictor_joint','predictor_zc','env_cls')
EXPECTED_HASH='969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de'
EPS=1e-12


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def require_file(path):
    if not path.is_file():raise FileNotFoundError(f'缺少已有文件：{path}。本步不会训练补齐。')
    with path.open('rb') as stream:
        if stream.read(140).startswith(b'version https://git-lfs.github.com/spec/'):
            raise ValueError(f'文件仍是Git LFS指针：{path}。请下载对应历史文件后再运行，不要重训。')


def load_fold(folder,held_out):
    """任务3 第16步：仅加载历史完整检查点和保存的来源训练数组，不拟合预处理。"""
    paths={n:folder/n for n in ('config.json','training.json','best_model.pt','splits.npz')}
    for path in paths.values():require_file(path)
    payload=json.loads(paths['config.json'].read_text(encoding='utf-8-sig'));cfg=TrainConfig(**payload['config'])
    if payload['input_sha256']!=EXPECTED_HASH or payload['feature_names']!=list(FEATURES):raise ValueError('历史数据或特征顺序不一致')
    if cfg.model!='v5' or cfg.environment_granularity!='region' or cfg.seed!=42:raise ValueError('必须使用历史完整地区V5基线')
    if cfg.lambda_graph!=0 or cfg.lambda_entropy!=0:raise ValueError('当前审计针对历史零图先验、零熵配置，不静默遗漏其他损失')
    if cfg.irm_mode!='v11' or cfg.independence!='hsic' or cfg.sparsity!='mean':raise ValueError('历史约束模式不匹配')
    if min(cfg.lambda_sparse,cfg.lambda_irm,cfg.lambda_env,cfg.lambda_independence)<=0:raise ValueError('完整基线的有效约束不能被关闭')
    report=json.loads(paths['training.json'].read_text(encoding='utf-8-sig'))
    selected=report['best_epoch'];full_epoch=max(cfg.sparse_start,cfg.irm_start)+cfg.ramp_epochs+1
    if selected<full_epoch or selected<cfg.first_selection_epoch():raise ValueError('检查点不在完整权重选模阶段')
    record=next(r for r in report['history'] if r['epoch']==selected)
    if record['invariant_ramp']!=1 or record['sparse_ramp']!=1:raise ValueError('历史选中轮次未达到完整约束')
    checkpoint=torch.load(paths['best_model.pt'],map_location='cpu',weights_only=True)
    if checkpoint['config']!=payload['config'] or checkpoint['feature_names']!=list(FEATURES):raise ValueError('模型与历史配置不匹配')
    env_map=checkpoint['env_map']
    if set(env_map)!=(set(REGIONS)-{held_out}) or sorted(env_map.values())!=[0,1]:raise ValueError('来源环境映射不匹配')
    model=models.CausalDisentangledModel(len(FEATURES),len(env_map),cfg.latent_dim,
        cfg.gate_init_std,gate_temperature=cfg.gate_temperature,joint=True).cpu()
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    if not np.allclose(model.gate.get_scores().numpy(),record['mask'],atol=2e-6,rtol=0):raise ValueError('模型门控与历史选中轮次不匹配')
    with np.load(paths['splits.npz'],allow_pickle=False) as archive:
        x=np.asarray(archive['X_train'],dtype=np.float32)
        y=np.asarray(archive['y_train'],dtype=np.float32).reshape(-1,1)
        env=np.asarray(archive['env_train']).astype(str)
        ids=archive['sample_id_train']
        if len(np.unique(ids))!=len(ids):raise ValueError('来源训练身份重复')
        if any(set(ids)&set(archive['sample_id_'+s]) for s in ('val','test')):raise ValueError('来源训练与其他划分重叠')
    if x.shape!=(len(y),len(FEATURES)) or len(env)!=len(y) or set(env)!=set(env_map):raise ValueError('来源数组与环境不匹配')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or set(np.unique(y))!={0.,1.}:raise ValueError('来源输入或标签无效')
    env=np.array([env_map[v] for v in env],dtype=np.int64)
    return model,cfg,(x,y,env),dict(selected_epoch=selected,source_n=len(y),
        artifact_sha256={name:digest(path) for name,path in paths.items()},
        source_sample_order_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest())


def parameter_groups(model):
    named=list(model.named_parameters());group={g:[] for g in GROUPS}
    for i,(name,p) in enumerate(named):
        prefix=name.split('.')[0]
        if prefix not in group:raise RuntimeError('遇到未审计参数组：'+prefix)
        group[prefix].append(i)
    if any(not v for v in group.values()):raise RuntimeError('参数组缺失')
    return named,group


def batch_terms(model,x,y,env,cfg):
    """任务3 第16步：沿用训练器目标与类权重，所有项使用同一批次的共享前向结果。"""
    logits,env_logits,zc,zs,mask=model(x)
    pos=torch.tensor([cfg.pos_weight])
    return dict(main_prediction=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos),
        zc_prediction=torch.nn.functional.binary_cross_entropy_with_logits(model.predictor_zc(zc),y,pos_weight=pos),
        sparse=cfg.lambda_sparse*mask.mean(),
        irm=cfg.lambda_irm*losses.irm_penalty(model,zc,y,env,cfg.irm_mode),
        environment=cfg.lambda_env*torch.nn.functional.cross_entropy(env_logits,env),
        independence=cfg.lambda_independence*losses.independence_loss(zc,zs,cfg.independence,sigma=cfg.hsic_sigma))


def audit_batch(model,x,y,env,cfg,context):
    """任务3 第16步：autograd.grad只返回梯度；无优化器、无参数更新。"""
    named,group=parameter_groups(model);params=tuple(p for _,p in named)
    terms=batch_terms(model,x,y,env,cfg);grads={};norms=[];relations=[];gate_rows=[]
    for term,value in terms.items():
        raw=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        grads[term]={g:torch.cat([(raw[i] if raw[i] is not None else torch.zeros_like(params[i])).detach().reshape(-1)
                                 for i in positions]) for g,positions in group.items()}
        for g,positions in group.items():
            vector=grads[term][g]
            if not torch.isfinite(vector).all():raise FloatingPointError('出现非有限诊断梯度')
            norms.append(dict(**context,loss=term,parameter_group=g,weighted_loss=float(value.detach()),
                gradient_norm=float(vector.norm()),connected_parameter_tensors=sum(raw[i] is not None for i in positions),
                parameter_tensors=len(positions)))
        for feature,gradient in zip(FEATURES,grads[term]['gate'].tolist()):
            gate_rows.append(dict(**context,loss=term,feature=feature,raw_gate_gradient=gradient,
                local_raw_gate_descent=-gradient))
    for g in GROUPS:
        classification=grads['main_prediction'][g]+grads['zc_prediction'][g]
        regularizers={name:grads[name][g] for name in REGULARIZERS[:-1]}
        regularizers['all_regularizers']=sum(regularizers.values())
        cn=float(classification.norm())
        for name,vector in regularizers.items():
            rn=float(vector.norm());dot=float(torch.dot(vector,classification));defined=cn>EPS and rn>EPS
            relations.append(dict(**context,loss=name,parameter_group=g,classification_gradient_norm=cn,
                regularizer_gradient_norm=rn,relative_norm=rn/cn if cn>EPS else None,
                cosine=float(np.clip(dot/(cn*rn),-1,1)) if defined else None,cosine_defined=defined,
                projection_on_classification=dot/(cn*cn) if cn>EPS else None))
    if any(p.grad is not None for p in params):raise RuntimeError('诊断意外写入了优化梯度')
    return norms,relations,gate_rows


def source_batches(env,batch_size,repeats,seed):
    """任务3 第16步：来源总体均匀抽样，不按目标表现或梯度值挑批次。"""
    if batch_size>len(env):raise ValueError('诊断批次大于来源样本数')
    rng=np.random.default_rng(seed)
    result=[]
    for _ in range(repeats):
        for attempt in range(100):
            ids=rng.choice(len(env),batch_size,replace=False)
            if all(np.sum(env[ids]==v)>=2 for v in np.unique(env)):break
        else:raise RuntimeError('无法得到包含两个来源环境的诊断批次')
        result.append(ids)
    return result


def summary_tables(norms,relations,gates):
    norm=norms.groupby(['held_out','mode','loss','parameter_group'],as_index=False).agg(
        mean_gradient_norm=('gradient_norm','mean'),std_gradient_norm=('gradient_norm','std'),
        mean_weighted_loss=('weighted_loss','mean'),connected_tensors=('connected_parameter_tensors','max'),batches=('batch','count'))
    relation=relations.groupby(['held_out','mode','loss','parameter_group'],as_index=False).agg(
        mean_relative_norm=('relative_norm','mean'),valid_relative_batches=('relative_norm','count'),
        mean_cosine=('cosine','mean'),valid_cosine_batches=('cosine','count'),
        negative_cosine_batches=('cosine',lambda v:int((v<0).sum())),
        mean_projection=('projection_on_classification','mean'))
    gate=gates.groupby(['held_out','mode','loss','feature'],as_index=False).agg(
        mean_local_descent=('local_raw_gate_descent','mean'),std_local_descent=('local_raw_gate_descent','std'),
        positive_direction_batches=('local_raw_gate_descent',lambda v:int((v>0).sum())),batches=('batch','count'))
    return norm,relation,gate


def draw(norm,relation,gate,output):
    """任务3 第16步：仅展示来源梯度，零连接与未定义余弦明确区分。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=norm[(norm.held_out==region)&(norm['mode']=='train_dropout')]
        matrix=part.pivot(index='loss',columns='parameter_group',values='mean_gradient_norm').reindex(index=TERMS,columns=GROUPS)
        positive=matrix.to_numpy()>0;logged=np.full(matrix.shape,np.nan);logged[positive]=np.log10(matrix.to_numpy()[positive])
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('lightgrey')
        ax.imshow(logged,aspect='auto',cmap=cmap,vmin=-8,vmax=1)
        ax.set_xticks(range(len(GROUPS)));ax.set_xticklabels(GROUPS,rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(len(TERMS)));ax.set_yticklabels(TERMS,fontsize=8);ax.set_title(region)
        for i in range(len(TERMS)):
            for j in range(len(GROUPS)):ax.text(j,i,f'{matrix.iloc[i,j]:.1e}',ha='center',va='center',fontsize=7,color='black' if not positive[i,j] else 'white')
    fig.suptitle('Weighted gradient norms; grey = zero; same log color scale; source batches only')
    fig.tight_layout();fig.savefig(output/'figures/01_loss_parameter_gradient_paths.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=relation[(relation.held_out==region)&(relation['mode']=='train_dropout')]
        values=part.pivot(index='loss',columns='parameter_group',values='mean_cosine').reindex(index=REGULARIZERS,columns=GROUPS)
        cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('lightgrey')
        im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=-1,vmax=1)
        ax.set_xticks(range(len(GROUPS)));ax.set_xticklabels(GROUPS,rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(len(REGULARIZERS)));ax.set_yticklabels(REGULARIZERS,fontsize=8);ax.set_title(region)
        for i in range(len(REGULARIZERS)):
            for j in range(len(GROUPS)):
                v=values.iloc[i,j];ax.text(j,i,f'{v:+.2f}' if np.isfinite(v) else 'N/A',ha='center',va='center',fontsize=8)
    fig.suptitle('Regularizer vs total classification gradient; grey = undefined, not no effect')
    fig.tight_layout();fig.savefig(output/'figures/02_regularizer_classification_directions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,9),sharey=True,layout='constrained')
    matrices=[]
    for mode in MODES:
        part=gate[(gate.held_out=='east')&(gate['mode']==mode)]
        matrices.append(part.pivot(index='feature',columns='loss',values='mean_local_descent').reindex(index=FEATURES,columns=TERMS))
    limit=max(1e-8,max(float(np.nanmax(np.abs(v.values))) for v in matrices))
    for ax,mode,values in zip(axes,MODES,matrices):
        im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(TERMS)));ax.set_xticklabels(TERMS,rotation=40,ha='right',fontsize=8)
        ax.set_yticks(range(len(FEATURES)));ax.set_yticklabels(FEATURES,fontsize=8)
        ax.set_title(f'East held out; source probes; {mode}')
    fig.suptitle('Local negative raw-gate gradient: positive points toward a larger zc gate; not importance')
    fig.colorbar(im,ax=axes,shrink=.7,label='Mean negative gradient of raw gate parameter')
    fig.savefig(output/'figures/03_source_gate_direction_by_feature.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=relation[(relation.held_out==region)&(relation.loss=='all_regularizers')]
        values=part.pivot(index='parameter_group',columns='mode',values='mean_relative_norm').reindex(index=GROUPS[:3],columns=MODES)
        values.plot.bar(ax=ax,rot=0);ax.set_title(region);ax.set_ylabel('Regularizers / classification gradient norm');ax.legend(fontsize=8)
    fig.suptitle('Local source-batch diagnostics at fixed checkpoints; not actual Adam update ratios')
    fig.tight_layout();fig.savefig(output/'figures/04_dropout_gradient_sensitivity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_2611/task3_step16_full_v5_loss_gradient_audit')
    parser.add_argument('--batches',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    if args.batches<2 or args.batch_size<4:raise ValueError('至少两个诊断批次，批次大小至少4')
    # 任务3 第16步：先检查全部文件，缺少任何历史模型都不自动重训。
    for region in REGIONS:
        for name in ('config.json','training.json','best_model.pt','splits.npz'):require_file(args.previous/region/'v5'/name)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    metadata=dict(completed=False,training_runs=0,optimizer_steps=0,model_loads=0,source_data_only=True,
        modes=MODES,batches=args.batches,batch_size=args.batch_size,input_sha256=EXPECTED_HASH,
        individual_records_exported=False,runner_sha256=digest(Path(__file__)),
        code_sha256={Path(m.__file__).name:digest(Path(m.__file__)) for m in (models,losses)},
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        prior_work_checked=['doc/实验2610.md','doc/实验2611.md','gate_step3','task1_step4_gate_alignment',
            'experiment3_feature_reliance','task3_step2_v5_representation_diagnostics',
            'task3_step3_fixed_representation_probes','task3_step14_reconstruction_objective_control',
            'task3_step15_full_v5_existing_artifact_audit'],
        limitations=['仅审计历史完整V5的最终检查点，不还原训练中的Adam动量或历史逐步梯度。',
            '负梯度余弦只说明局部方向关系，不证明损害东部泛化或识别因果特征。',
            '梯度范数跨不同参数坐标的比较受尺度影响，不等于优化器真实更新量。',
            'train_dropout仅启用Dropout用于只读求导；不优化参数，eval作为随机失活敏感性对照。',
            '诊断批次是同一来源人群的重复抽样，不是独立人群或跨初始化重复。',
            '门控的局部负梯度不等于特征重要性，也不代表目标地区的因果作用。'])
    dump(args.output/'experiment_metadata.json',metadata)
    all_norm=[];all_relation=[];all_gate=[];checks=[]
    for index,region in enumerate(REGIONS):
        with torch.random.fork_rng(devices=[]):
            model,cfg,(x,y,env),info=load_fold(args.previous/region/'v5',region)
            before=state_hash(model);batches=source_batches(env,args.batch_size,args.batches,26111600+index)
            h=hashlib.sha256()
            for ids in batches:h.update(ids.astype(np.int64).tobytes())
            for mode in MODES:
                model.train(mode=='train_dropout')
                for repeat,ids in enumerate(batches):
                    torch.manual_seed(261116000+index*1000+repeat)
                    context=dict(held_out=region,mode=mode,batch=repeat)
                    values=audit_batch(model,torch.as_tensor(x[ids]),torch.as_tensor(y[ids]),torch.as_tensor(env[ids]),cfg,context)
                    for collection,rows in zip((all_norm,all_relation,all_gate),values):collection.extend(rows)
            after=state_hash(model)
            if before!=after:raise RuntimeError('只读审计意外改变模型状态')
            model.eval()
            checks.append(dict(held_out=region,selected_epoch=info['selected_epoch'],source_n=info['source_n'],
                model_state_unchanged=True,optimizer_gradients_untouched=all(p.grad is None for p in model.parameters()),
                common_mode_batch_sha256=h.hexdigest(),state_sha256=before))
            metadata.setdefault('fold_artifacts',{})[region]=info
        metadata['model_loads']+=1
        print(f'任务3 第16步：已完成{region}完整V5只读梯度审计，参数未更新',flush=True)
    norm=pd.DataFrame(all_norm);relation=pd.DataFrame(all_relation);gate=pd.DataFrame(all_gate)
    ns,rs,gs=summary_tables(norm,relation,gate)
    for name,values in dict(gradient_norms=norm,gradient_relations=relation,gate_local_directions=gate,
        gradient_norm_summary=ns,gradient_relation_summary=rs,gate_direction_summary=gs,control_checks=pd.DataFrame(checks)).items():
        values.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(ns,rs,gs,args.output)
    metadata['completed']=True;dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第16步完成：训练0次，优化0步，加载3个已有模型。输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
