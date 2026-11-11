"""任务3 第12步：控制分类信息路径，检查双块分工是否限制低维表格预测。

共同初始参数、起始预测、来源抽样和训练政策；只改变分类可访问的信息。
固定0/0.1条件MMD；原始19维数值编码不变。不是论文原样复现或因果识别。
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sparse_invariance_feature_pilot as base
import two_block_information_audit as audit
from elderly_health.data import prepare_splits
from elderly_health import conditional_mmd_adapter as adapter

REGIONS=('east','central','west')
SEEDS=(42,43,44)
ROUTES=('prediction_only','joint','raw_joint')
MMD_WEIGHTS=(0.,.1)
SETTINGS={k:v for k,v in adapter.SETTINGS.items() if k!='seed'}
CONTRASTS={'joint_minus_prediction':('joint','prediction_only'),
           'raw_joint_minus_joint':('raw_joint','joint'),
           'raw_joint_minus_prediction':('raw_joint','prediction_only')}


def model_hash(model, core_only=False):
    """任务3 第12步：记录共同初值；额外路径的零初值也纳入完整哈希。"""
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if core_only and name.startswith(('residual_head.','raw_head.')):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_model(seed,input_dim):
    """任务3 第12步：先按历史顺序初始化，再增加零初值路径；所有组参数总数相同。"""
    torch.manual_seed(seed)
    model=adapter.build_model(input_dim)
    model['residual_head']=torch.nn.Linear(8,1,bias=False)
    model['raw_head']=torch.nn.Linear(input_dim,1,bias=False)
    torch.nn.init.zeros_(model['residual_head'].weight)
    torch.nn.init.zeros_(model['raw_head'].weight)
    return model


def forward(model,x,route):
    """任务3 第12步：同样计算三条路径，以固定开关限制其进入分类分数。"""
    if route not in ROUTES:raise ValueError('未知分类路径')
    a=model['prediction_encoder'](x);b=model['residual_encoder'](x)
    p=model['classifier'](a)
    r=model['residual_head'](b)*(route in ('joint','raw_joint'))
    raw=model['raw_head'](x)*(route=='raw_joint')
    return p+r+raw,a,b,(p,r,raw)


def predict(model,x,route):
    model.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            z,_,_,_=forward(model,torch.as_tensor(x[start:start+256]),route)
            rows.append(torch.sigmoid(z).numpy().ravel())
    return np.concatenate(rows)


def fit(train,val,seed,route,mmd_weight,settings):
    """任务3 第12步：仅来源数据参与优化与选模，保持历史地区等量抽样政策。"""
    adapter.verify_source_splits(train,val)
    domains,groups,conditional=adapter.training_groups(train)
    model=build_model(seed,train.x.shape[1]);initial=model_hash(model)
    model.eval()
    initial_probability_hash=hashlib.sha256(predict(model,train.x[:64],route).tobytes()).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([settings['positive_weight']]))
    batch_rng=torch.Generator().manual_seed(seed);mmd_rng=torch.Generator().manual_seed(seed+1)
    sample_hash=hashlib.sha256();prefixes={}
    x_all=torch.as_tensor(train.x);y_all=torch.as_tensor(train.y[:,None])
    steps=math.ceil(len(train.x)/(len(domains)*settings['per_environment_batch_size']))
    history=[];best=None;best_auc=-np.inf;best_epoch=0;stale=0
    for epoch in range(1,settings['epochs']+1):
        model.train();totals=np.zeros(3)
        for _ in range(steps):
            ids=np.concatenate([groups[d][torch.randint(len(groups[d]),(settings['per_environment_batch_size'],),generator=batch_rng).numpy()] for d in domains])
            sample_hash.update(ids.astype(np.int64).tobytes())
            x,y=x_all[ids],y_all[ids]
            logit,a,b,_=forward(model,x,route)
            classification=criterion(logit,y)
            reconstruction=(model['decoder'](torch.cat([a,b],dim=1))-x).square().mean()
            penalties=[]
            for label in (0,1):
                blocks=[]
                for domain in domains:
                    positions=conditional[(domain,label)]
                    chosen=positions[torch.randint(len(positions),(settings['mmd_per_class_environment'],),generator=mmd_rng).numpy()]
                    sample_hash.update(chosen.astype(np.int64).tobytes())
                    blocks.append(model['prediction_encoder'](x_all[chosen]))
                penalties.append(adapter.mmd_squared(*blocks))
            penalty=torch.stack(penalties).mean()
            loss=classification+settings['reconstruction_weight']*reconstruction+mmd_weight*penalty
            if not torch.isfinite(loss):raise FloatingPointError('出现非有限训练损失')
            optimizer.zero_grad();loss.backward();optimizer.step()
            totals += [classification.item(),reconstruction.item(),penalty.item()]
        source_auc=float(roc_auc_score(val.y,predict(model,val.x,route)))
        prefixes[str(epoch)]=sample_hash.hexdigest()
        eligible=epoch>=settings['selection_start_epoch']
        history.append(dict(epoch=epoch,source_validation_auc=source_auc,eligible_for_selection=eligible,
            classification_loss=float(totals[0]/steps),reconstruction_loss=float(totals[1]/steps),conditional_mmd=float(totals[2]/steps)))
        if eligible:
            if source_auc>best_auc:
                best_auc=source_auc;best_epoch=epoch;stale=0;best=copy.deepcopy(model.state_dict())
            else:stale+=1
        if epoch==1 or epoch%30==0:
            print(f'任务3 第12步：种子{seed}，路径{route}，约束{mmd_weight}，轮次{epoch}，来源验证AUC={source_auc:.4f}',flush=True)
        if eligible and stale>=settings['patience']:break
    if best is None:raise RuntimeError('没有可选来源检查点')
    model.load_state_dict(best);model.eval()
    return model,dict(seed=seed,route=route,mmd_weight=mmd_weight,best_epoch=best_epoch,best_source_auc=best_auc,
        initial_state_sha256=initial,initial_probability_sha256=initial_probability_hash,
        sampling_prefix_sha256=prefixes,history=history,parameter_count=sum(p.numel() for p in model.parameters()))


def controls(reports):
    """任务3 第12步：六条件必须有共同初始状态、预测、参数数目与共同轮次抽样。"""
    if len(reports)!=len(ROUTES)*len(MMD_WEIGHTS):raise RuntimeError('实验条件不完整')
    for key in ('initial_state_sha256','initial_probability_sha256','parameter_count'):
        if len({r[key] for r in reports})!=1:raise RuntimeError(f'控制条件不一致：{key}')
    common=set.intersection(*(set(r['sampling_prefix_sha256']) for r in reports))
    if not common or any(len({r['sampling_prefix_sha256'][e] for r in reports})!=1 for e in common):
        raise RuntimeError('共同轮次抽样不一致')
    return dict(same_initial_state=True,same_initial_prediction=True,same_parameter_count=True,
                common_sampling_epochs=len(common),sampling_matched=True)


def diagnostics(model,frame,split,route,context,stage):
    """任务3 第12步：表示秩和各路径分数波动；分数波动不等于特征重要性。"""
    model.eval()
    with torch.no_grad():
        z,a,b,terms=forward(model,torch.as_tensor(split.x),route)
        reconstruction=model['decoder'](torch.cat([a,b],dim=1))
        rec=float((reconstruction-torch.as_tensor(split.x)).square().mean())
    rows=[audit.representation_stats(context,stage,name,values.numpy()) for name,values in
          (('prediction8',a),('residual8',b),('joint16',torch.cat([a,b],dim=1)))]
    contribution=dict(**context,stage=stage,reconstruction_mse=rec,
        prediction_logit_std=float(terms[0].std(unbiased=False)),residual_logit_std=float(terms[1].std(unbiased=False)),
        raw_logit_std=float(terms[2].std(unbiased=False)),total_logit_std=float(z.std(unbiased=False)))
    return rows,contribution


def differences(table):
    """任务3 第12步：同种子同人群配对，保留所有比较，不按目标表现选条件。"""
    keys=['held_out','seed','mmd_weight','stage','group','subgroup']
    rows=[]
    for name,(after,before) in CONTRASTS.items():
        a=table[table.route==after];b=table[table.route==before]
        merged=a.merge(b,on=keys,suffixes=('_after','_before'),validate='one_to_one')
        if len(merged)!=len(a) or len(merged)!=len(b):raise RuntimeError('配对比较缺行')
        out=merged[keys].copy();out['contrast']=name
        for metric in ('auc','auprc','brier'):
            out[metric+'_change']=merged[metric+'_after']-merged[metric+'_before']
        rows.append(out)
    return pd.concat(rows,ignore_index=True)


def macro_table(table):
    sites=table[(table.stage=='test')&(table.group!='REGION_ALL')]
    region=sites.groupby(['held_out','seed','mmd_weight','route','subgroup'],as_index=False)[['auc','auprc','brier']].mean()
    global_=sites.groupby(['seed','mmd_weight','route','subgroup'],as_index=False)[['auc','auprc','brier']].mean()
    global_['held_out']='ALL'
    return pd.concat([region,global_],ignore_index=True)


def draw(table,macro,delta,representation,path_use,output):
    """任务3 第12步：展示地区与站点收益、指标取舍和信息路径利用。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,weight in zip(axes,MMD_WEIGHTS):
        part=macro[(macro.mmd_weight==weight)&(macro.subgroup=='all')]
        values=part.groupby(['held_out','route']).auc.mean().unstack().reindex(index=[*REGIONS,'ALL'],columns=ROUTES)
        values.plot.bar(ax=ax,rot=0);ax.set_title(f'Conditional MMD weight = {weight}');ax.set_ylabel('Equal-site macro AUC');ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(output/'figures/01_pathway_region_auc.png',dpi=180);plt.close(fig)
    site=delta[(delta.stage=='test')&(delta.subgroup=='all')&(delta.group!='REGION_ALL')]
    order=[(r,str(s)) for r in REGIONS for s in sorted(site.loc[site.held_out==r,'group'].unique())]
    matrices=[]
    for weight in MMD_WEIGHTS:
        values=site[site.mmd_weight==weight].groupby(['held_out','group','contrast']).auc_change.mean().unstack()
        matrices.append(values.reindex(index=pd.MultiIndex.from_tuples(order),columns=CONTRASTS))
    limit=max(.01,max(float(np.nanmax(np.abs(m.values))) for m in matrices))
    fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,weight,values in zip(axes,MMD_WEIGHTS,matrices):
        im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(3));ax.set_xticklabels(['Joint - prediction','Raw+joint - joint','Raw+joint - prediction'],rotation=20,ha='right',fontsize=8)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([f'{r}: {s}' for r,s in order]);ax.set_title(f'MMD = {weight}; mean across seeds')
        for i in range(len(order)):
            for j in range(3):
                value=values.iloc[i,j];ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.65*limit else 'black')
    fig.subplots_adjust(left=.16,right=.86,wspace=.1,bottom=.16,top=.94)
    cax=fig.add_axes([.9,.2,.015,.6]);fig.colorbar(im,cax=cax,label='Paired site AUC change')
    fig.savefig(output/'figures/02_site_pathway_changes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,metric in zip(axes,('auc','auprc','brier')):
        for weight in MMD_WEIGHTS:
            part=macro[(macro.mmd_weight==weight)&(macro.subgroup=='all')]
            pivot=part.pivot(index=['held_out','seed'],columns='route',values=metric)
            for j,route in enumerate(('joint','raw_joint')):
                values=(pivot[route]-pivot.prediction_only).groupby(level='held_out').mean().reindex(REGIONS)
                ax.plot(range(3),values,'o-',label=f'{route}, MMD={weight}',alpha=.8)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3));ax.set_xticklabels(REGIONS)
        ax.set_title(metric.upper()+(' (lower is better)' if metric=='brier' else ' (higher is better)'))
        ax.set_ylabel('Change from prediction-only');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/03_pathway_metric_tradeoffs.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    rep=representation[(representation.stage=='test')&(representation.space=='prediction8')].copy()
    rep.groupby(['route','mmd_weight']).effective_rank.mean().unstack().reindex(ROUTES).plot.bar(ax=axes[0],rot=0)
    axes[0].set_ylabel('Prediction-block effective rank')
    axes[0].set_title('Mean over regions and seeds; rank alone is not collapse')
    use=path_use[(path_use.stage=='test')&(path_use.held_out=='east')].groupby(['route','mmd_weight'])[['prediction_logit_std','residual_logit_std','raw_logit_std']].mean()
    use=use.reindex(pd.MultiIndex.from_product([ROUTES,MMD_WEIGHTS],names=['route','mmd_weight']))
    use.plot.bar(ax=axes[1],rot=25);axes[1].set_title('East: component score variation');axes[1].set_ylabel('Component logit std (not importance)')
    axes[1].set_xticklabels([f'{r} / {w:g}' for r in ROUTES for w in MMD_WEIGHTS],fontsize=8)
    axes[1].set_xlabel('Classification path / MMD weight')
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/04_representation_and_path_use.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--output',type=Path,default=base.ROOT/'outputs/analysis_2611/task3_step12_two_block_pathway_control')
    args=parser.parse_args()
    if audit.digest(args.data)!=base.EXPECTED_HASH:raise ValueError('固定数据版本不一致')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame=base.attach_regions(base.read_raw(args.data))
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    metadata=dict(completed=False,settings=SETTINGS,split_seed=42,seeds=SEEDS,routes=ROUTES,mmd_weights=MMD_WEIGHTS,
        input_sha256=base.EXPECTED_HASH,expected_training_runs=len(REGIONS)*len(SEEDS)*len(ROUTES)*len(MMD_WEIGHTS),
        target_used_for_training_or_selection=False,individual_records_exported=False,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        code_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (base,adapter,audit)},runner_sha256=audit.digest(Path(__file__)),
        limitations=['本步检验结构策略，不是因果识别或新论文复现；三种子不是独立人群。',
        '共同参数总数与起始预测受控，但有效分类容量及梯度路径随信息开关改变，不能单独解释成信息量效应。',
        'MMD只作用于预测块；允许残余块或原输入分类时，这些路径没有相同的不变性保证。',
        '重建保留历史数值编码与MSE，类别编码距离没有临床含义。',
        '来源验证重复用于选模，目标地区已经多次探索；仅报告全部条件的配对变化。',
        '分类路径分数标准差不是独立贡献或因果重要性；表示有效秩小也不直接证明崩溃。'])
    base.dump(args.output/'experiment_metadata.json',metadata)
    rows=[];reports=[];checks=[];rep=[];path_use=[]
    for region in REGIONS:
        frames=base.held_out_split(frame,region,seed=42,column=base.REGION)
        splits,prep=prepare_splits(frames,environment_column=base.REGION)
        if any(set(a.sample_ids)&set(b.sample_ids) for i,a in enumerate(splits) for b in splits[i+1:]):raise RuntimeError('人员划分重叠')
        for seed in SEEDS:
            fitted={};local_reports=[]
            for weight in MMD_WEIGHTS:
                for route in ROUTES:
                    print(f'任务3 第12步：留出{region}，初始化{seed}，分类路径{route}，约束{weight}',flush=True)
                    model,report=fit(splits[0],splits[1],seed,route,weight,SETTINGS)
                    fitted[(route,weight)]=model;local_reports.append(report)
                    folder=args.output/region/str(seed)/f'mmd_{weight:g}'/route;folder.mkdir(parents=True)
                    base.dump(folder/'training.json',report)
                    torch.save(dict(state_dict=model.state_dict(),settings=SETTINGS,seed=seed,route=route,mmd_weight=weight,features=list(base.FEATURES)),folder/'model.pt')
                    np.savez(folder/'preprocessing.npz',median=prep.imputer.statistics_,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
                    reports.append(dict(held_out=region,seed=seed,route=route,mmd_weight=weight,best_epoch=report['best_epoch'],source_validation_auc=report['best_source_auc']))
            checks.append(dict(held_out=region,seed=seed,**controls(local_reports)))
            # 任务3 第12步：同种子的六组来源选模全部结束后，才评估留出地区。
            for (route,weight),model in fitted.items():
                context=dict(held_out=region,seed=seed,route=route,mmd_weight=weight)
                for stage,part,split in zip(('source_train','source_validation','test'),frames,splits):
                    rows.extend(audit.evaluate(context,stage,part,predict(model,split.x,route)))
                    if stage!='source_train':
                        values,contribution=diagnostics(model,part,split,route,context,stage)
                        rep.extend(values);path_use.append(contribution)
            for name,values in dict(metrics=rows,source_only_selection=reports,control_checks=checks,representation_diagnostics=rep,pathway_diagnostics=path_use).items():
                pd.DataFrame(values).to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
            metadata['completed_training_runs']=len(reports);base.dump(args.output/'experiment_metadata.json',metadata)
    table=pd.DataFrame(rows);macro=macro_table(table);delta=differences(table)
    macro.to_csv(args.output/'region_and_global_macro.csv',index=False,encoding='utf-8-sig')
    delta.to_csv(args.output/'matched_seed_differences.csv',index=False,encoding='utf-8-sig')
    summary=delta[delta.stage=='test'].groupby(['held_out','mmd_weight','group','subgroup','contrast'],as_index=False).agg(
        mean_auc_change=('auc_change','mean'),min_auc_change=('auc_change','min'),max_auc_change=('auc_change','max'),
        positive_seed_count=('auc_change',lambda x:int((x>0).sum())),valid_seed_count=('auc_change','count'),
        mean_auprc_change=('auprc_change','mean'),mean_brier_change=('brier_change','mean'))
    summary.to_csv(args.output/'seed_stability_summary.csv',index=False,encoding='utf-8-sig')
    draw(table,macro,delta,pd.DataFrame(rep),pd.DataFrame(path_use),args.output)
    metadata['completed']=True;base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第12步完成：训练{len(reports)}次，输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
