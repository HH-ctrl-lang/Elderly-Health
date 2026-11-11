"""任务3 第14步：低维输入的重建任务消融与来源梯度方向诊断。

固定预测块分类、原始19维数值编码、MMD权重0/0.1；只开关重建损失。
四组均训练满300轮，从第1轮起按来源合并验证AUC选模，不使用目标地区选择。
不是论文原样复现，也不把梯度冲突、有效秩或重建准确认定为因果证据。
"""
import argparse
import copy
import hashlib
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
import two_block_pathway_control as pathway
from two_block_pathway_control import base,audit,adapter,build_model,forward,predict,model_hash
from elderly_health.data import prepare_splits

REGIONS=pathway.REGIONS
SEEDS=pathway.SEEDS
MMD_WEIGHTS=pathway.MMD_WEIGHTS
RECONSTRUCTION_WEIGHTS=(0.,1.)
SETTINGS={k:v for k,v in pathway.SETTINGS.items()
          if k not in ('selection_start_epoch','patience','reconstruction_weight')}
PROBE_PER_SOURCE_REGION=32


def source_probe_indices(train,seed):
    """任务3 第14步：独立随机数选固定来源探针，不改变优化抽样或读取目标人群。"""
    domains,groups,_=adapter.training_groups(train)
    rng=np.random.default_rng(26111400+seed)
    return np.concatenate([rng.choice(groups[d],size=min(PROBE_PER_SOURCE_REGION,len(groups[d])),replace=False)
                           for d in domains])


def gradient_probe(model,x,y,reconstruction_weight,settings):
    """任务3 第14步：只测预测编码器上分类与重建梯度，不回传到优化器。"""
    model.eval()
    with torch.enable_grad():
        _,a,b,_=forward(model,x,'prediction_only')
        classification=torch.nn.functional.binary_cross_entropy_with_logits(model['classifier'](a),y,
            pos_weight=torch.tensor([settings['positive_weight']]))
        reconstruction=(model['decoder'](torch.cat([a,b],dim=1))-x).square().mean()
        params=tuple(model['prediction_encoder'].parameters())
        gc=torch.autograd.grad(classification,params,retain_graph=True)
        gr=torch.autograd.grad(reconstruction,params)
        gc=torch.cat([g.detach().reshape(-1) for g in gc]);gr=torch.cat([g.detach().reshape(-1) for g in gr])
        nc=float(gc.norm());nr=float(gr.norm());dot=float(torch.dot(gc,gr))
    valid=nc>1e-12 and nr>1e-12
    return dict(classification_gradient_norm=nc,reconstruction_gradient_norm=nr,
        cosine_defined=valid,gradient_cosine=float(np.clip(dot/(nc*nr),-1.,1.)) if valid else None,
        weighted_reconstruction_to_classification_norm=reconstruction_weight*nr/nc if nc>1e-12 else None,
        reconstruction_gradient_applied=reconstruction_weight>0,
        probe_classification_loss=float(classification.detach()),probe_reconstruction_mse=float(reconstruction.detach()))


def fit(train,val,seed,mmd_weight,reconstruction_weight,settings):
    """任务3 第14步：相同固定预算内按来源验证AUC选模，目标地区不进入函数。"""
    route='prediction_only'
    if reconstruction_weight not in RECONSTRUCTION_WEIGHTS:raise ValueError('未知重建条件')
    if settings['epochs']<1:raise ValueError('训练轮数必须为正')
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
    probe_ids=source_probe_indices(train,seed)
    probe_x=x_all[probe_ids];probe_y=y_all[probe_ids]
    probe_hash=hashlib.sha256(probe_ids.astype(np.int64).tobytes()).hexdigest()
    history=[];gradient_rows=[];best=None;best_auc=-np.inf;best_epoch=0
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
            loss=classification+reconstruction_weight*reconstruction+mmd_weight*penalty
            if not torch.isfinite(loss):raise FloatingPointError('出现非有限训练损失')
            optimizer.zero_grad();loss.backward();optimizer.step()
            totals += [classification.item(),reconstruction.item(),penalty.item()]
        source_auc=float(roc_auc_score(val.y,predict(model,val.x,route)))
        prefixes[str(epoch)]=sample_hash.hexdigest()
        history.append(dict(epoch=epoch,source_validation_auc=source_auc,
            classification_loss=float(totals[0]/steps),reconstruction_loss=float(totals[1]/steps),conditional_mmd=float(totals[2]/steps)))
        if not np.isfinite(source_auc):raise FloatingPointError('来源验证AUC非有限')
        if source_auc>best_auc:
            best_auc=source_auc;best_epoch=epoch;best=copy.deepcopy(model.state_dict())
        if epoch==1 or epoch%30==0 or epoch==settings['epochs']:
            gradient_rows.append(dict(epoch=epoch,stage='training_probe',
                **gradient_probe(model,probe_x,probe_y,reconstruction_weight,settings)))
        if epoch==1 or epoch%30==0:
            print(f'任务3 第14步：种子{seed}，重建{reconstruction_weight}，约束{mmd_weight}，轮次{epoch}，来源验证AUC={source_auc:.4f}',flush=True)
    if best is None:raise RuntimeError('没有可选来源模型')
    model.load_state_dict(best);model.eval()
    gradient_rows.append(dict(epoch=best_epoch,stage='selected_probe',
        **gradient_probe(model,probe_x,probe_y,reconstruction_weight,settings)))
    report=dict(seed=seed,route=route,mmd_weight=mmd_weight,reconstruction_weight=reconstruction_weight,
        initial_state_sha256=initial,initial_probability_sha256=initial_probability_hash,
        sampling_prefix_sha256=prefixes,history=history,trained_epochs=len(history),
        parameter_count=sum(p.numel() for p in model.parameters()),probe_sample_sha256=probe_hash,
        best_epoch=best_epoch,best_source_auc=best_auc,selected_state_sha256=model_hash(model),
        gradient_diagnostics=gradient_rows)
    expected=max(history,key=lambda r:r['source_validation_auc'])
    if expected['epoch']!=best_epoch or expected['source_validation_auc']!=best_auc:
        raise RuntimeError('来源选模与日志最大值不一致')
    return model,report


def control_checks(reports,settings):
    """任务3 第14步：四条件必须有相同初值、预测、参数总数、探针及每轮抽样。"""
    expected={(m,r) for m in MMD_WEIGHTS for r in RECONSTRUCTION_WEIGHTS}
    if len(reports)!=len(expected) or {(r['mmd_weight'],r['reconstruction_weight']) for r in reports}!=expected:
        raise RuntimeError('四组条件不完整或重复')
    for key in ('initial_state_sha256','initial_probability_sha256','parameter_count','probe_sample_sha256'):
        if len({r[key] for r in reports})!=1:raise RuntimeError(f'控制条件不一致：{key}')
    epochs={str(e) for e in range(1,settings['epochs']+1)}
    if any(set(r['sampling_prefix_sha256'])!=epochs or r['trained_epochs']!=settings['epochs'] for r in reports):
        raise RuntimeError('未完成全部固定预算或轮次缺失')
    if any(len({r['sampling_prefix_sha256'][e] for r in reports})!=1 for e in epochs):
        raise RuntimeError('四条件来源抽样不一致')
    for r in reports:
        best=max(r['history'],key=lambda row:row['source_validation_auc'])
        if best['epoch']!=r['best_epoch'] or best['source_validation_auc']!=r['best_source_auc']:
            raise RuntimeError('来源选模没有保留最早的AUC最大值')
    return dict(same_initial_state=True,same_initial_prediction=True,same_parameter_count=True,
        same_probe_sample=True,sampling_matched=True,common_sampling_epochs=len(epochs),
        fixed_budget_matched=True,source_argmax_verified=True)


def paired_differences(table,keys):
    """任务3 第14步：同地区、种子与MMD配对，正数表示取消重建后指标增加。"""
    no=table[table.reconstruction_weight==0.];yes=table[table.reconstruction_weight==1.]
    merged=no.merge(yes,on=keys,suffixes=('_no_reconstruction','_reconstruction'),validate='one_to_one')
    if len(merged)!=len(no) or len(merged)!=len(yes):raise RuntimeError('重建消融配对缺行')
    result=merged[keys].copy()
    for metric in ('auc','auprc','brier'):
        for suffix in ('no_reconstruction','reconstruction'):
            result[metric+'_'+suffix]=merged[metric+'_'+suffix]
        result[metric+'_change']=merged[metric+'_no_reconstruction']-merged[metric+'_reconstruction']
    return result


def macro_table(table):
    sites=table[(table.stage=='test')&(table.group!='REGION_ALL')]
    keys=['seed','mmd_weight','reconstruction_weight','subgroup']
    region=sites.groupby(['held_out']+keys,as_index=False)[['auc','auprc','brier']].mean()
    global_=sites.groupby(keys,as_index=False)[['auc','auprc','brier']].mean();global_['held_out']='ALL'
    return pd.concat([region,global_],ignore_index=True)


def draw(macro,delta,curves,gradients,output):
    """任务3 第14步：来源曲线、地区指标、站点稳定性与训练梯度方向。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={0.:'tab:blue',1.:'tab:orange'}
    names={0.:'No reconstruction',1.:'With reconstruction'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for row,weight in enumerate(MMD_WEIGHTS):
        for ax,region in zip(axes[row],REGIONS):
            for reconstruction in RECONSTRUCTION_WEIGHTS:
                part=curves[(curves.mmd_weight==weight)&(curves.reconstruction_weight==reconstruction)&(curves.held_out==region)]
                values=part.groupby('epoch').source_validation_auc.agg(['mean','min','max'])
                ax.plot(values.index,values['mean'],color=colors[reconstruction],label=names[reconstruction])
                ax.fill_between(values.index,values['min'],values['max'],color=colors[reconstruction],alpha=.12)
            ax.set_title(f'{region}, MMD={weight:g}');ax.set_xlabel('Training epoch')
            ax.set_ylabel('Source validation AUC');ax.legend(fontsize=8)
    fig.suptitle('Same 300-epoch budget; mean and seed range (not confidence interval)')
    fig.tight_layout();fig.savefig(output/'figures/01_source_reconstruction_learning.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,weight in enumerate(MMD_WEIGHTS):
        for ax,metric in zip(axes[row],('auc','auprc','brier')):
            part=macro[(macro.mmd_weight==weight)&(macro.subgroup=='all')]
            values=part.groupby(['held_out','reconstruction_weight'])[metric].mean().unstack().reindex([*REGIONS,'ALL'])
            values.reindex(columns=RECONSTRUCTION_WEIGHTS).rename(columns=names).plot.bar(ax=ax,rot=0)
            ax.set_title(f'MMD={weight:g}: {metric.upper()} '+('(lower better)' if metric=='brier' else '(higher better)'))
            ax.set_ylabel('Equal-site macro');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/02_region_reconstruction_tradeoffs.png',dpi=180);plt.close(fig)
    site=delta[(delta.stage=='test')&(delta.subgroup=='all')&(delta.group!='REGION_ALL')]
    order=[(r,str(s)) for r in REGIONS for s in sorted(site.loc[site.held_out==r,'group'].unique())]
    matrices=[]
    for weight in MMD_WEIGHTS:
        values=site[site.mmd_weight==weight].pivot(index=['held_out','group'],columns='seed',values='auc_change')
        values=values.reindex(index=pd.MultiIndex.from_tuples(order),columns=SEEDS)
        values['Mean']=values.mean(axis=1);matrices.append(values)
    limit=max(.01,max(float(np.nanmax(np.abs(v.values))) for v in matrices))
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True)
    for ax,weight,values in zip(axes,MMD_WEIGHTS,matrices):
        im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(values.columns)));ax.set_xticklabels(values.columns)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([f'{r}: {s}' for r,s in order])
        ax.set_title(f'MMD={weight:g}: no reconstruction minus reconstruction')
        for i in range(len(values)):
            for j in range(len(values.columns)):
                v=values.iloc[i,j];ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(v)>.65*limit else 'black')
    fig.subplots_adjust(left=.17,right=.86,wspace=.12,bottom=.08,top=.94)
    cax=fig.add_axes([.90,.18,.015,.65]);fig.colorbar(im,cax=cax,label='Paired site AUC change')
    fig.savefig(output/'figures/03_site_reconstruction_seed_changes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for col,weight in enumerate(MMD_WEIGHTS):
        for reconstruction in RECONSTRUCTION_WEIGHTS:
            part=gradients[(gradients.stage=='training_probe')&(gradients.mmd_weight==weight)&
                           (gradients.reconstruction_weight==reconstruction)]
            values=part.groupby('epoch')[['gradient_cosine','weighted_reconstruction_to_classification_norm']].mean()
            axes[0,col].plot(values.index,values.gradient_cosine,'o-',color=colors[reconstruction],label=names[reconstruction])
            axes[1,col].plot(values.index,values.weighted_reconstruction_to_classification_norm,'o-',
                             color=colors[reconstruction],label=names[reconstruction])
        axes[0,col].axhline(0,color='black',lw=.8);axes[0,col].set_ylim(-1,1)
        axes[0,col].set_title(f'MMD={weight:g}; source-only probe')
        axes[0,col].set_ylabel('Classification vs reconstruction gradient cosine')
        axes[1,col].set_ylabel('Weighted reconstruction / classification norm')
        axes[1,col].set_xlabel('Training epoch')
        for ax in axes[:,col]:ax.legend(fontsize=8)
    fig.suptitle('Prediction encoder only; mean over regions and seeds; cosine alone is not harm')
    fig.tight_layout();fig.savefig(output/'figures/04_source_gradient_directions.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--output',type=Path,default=base.ROOT/'outputs/analysis_2611/task3_step14_reconstruction_objective_control')
    args=parser.parse_args()
    if audit.digest(args.data)!=base.EXPECTED_HASH:raise ValueError('固定数据版本不一致')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame=base.attach_regions(base.read_raw(args.data))
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    metadata=dict(completed=False,settings=SETTINGS,split_seed=42,seeds=SEEDS,mmd_weights=MMD_WEIGHTS,
        reconstruction_weights=RECONSTRUCTION_WEIGHTS,route='prediction_only',selection_start_epoch=1,
        selection_metric='pooled_source_validation_auc',stopping_rule='fixed_epoch_budget_no_early_stopping',
        probe_per_source_region=PROBE_PER_SOURCE_REGION,input_sha256=base.EXPECTED_HASH,
        expected_training_runs=len(REGIONS)*len(SEEDS)*len(MMD_WEIGHTS)*len(RECONSTRUCTION_WEIGHTS),
        target_used_for_training_or_selection=False,individual_records_exported=False,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        runner_sha256=audit.digest(Path(__file__)),
        code_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (pathway,base,audit,adapter)},
        limitations=['只检验重建目标开关，不能据此认定某个输入特征有因果作用。',
            '两组分类路径、有效预测结构及参数总数相同；取消重建后残余块与解码器不再承担有效训练任务。',
            '重建沿用标准化数值输入的MSE，类别编号间距离没有临床意义；不能把重建误差直接解释为医学信息量。',
            '不同于第13步的晚期早停，本步四条件统一完整300轮，候选检查点范围完全相同。',
            '梯度诊断只用固定来源训练样本；负余弦只表示局部方向冲突，不能单独证明外部性能下降。',
            '取消重建组的原始重建梯度是假设诊断，不实际加入训练；其解码器质量不与有重建组等同。',
            '来源验证反复使用，目标地区已多次探索；三初始化不是独立人群，不作确认性显著性结论。'])
    base.dump(args.output/'experiment_metadata.json',metadata)
    rows=[];selection=[];checks=[];curves=[];gradients=[];rep=[];path_use=[];runs=0
    for region in REGIONS:
        frames=base.held_out_split(frame,region,seed=42,column=base.REGION)
        splits,prep=prepare_splits(frames,environment_column=base.REGION)
        if any(set(a.sample_ids)&set(b.sample_ids) for i,a in enumerate(splits) for b in splits[i+1:]):raise RuntimeError('人员划分重叠')
        manifest={name:dict(n=len(split.y),positive_count=int(split.y.sum()),
            sample_ids_sha256=hashlib.sha256(np.asarray(split.sample_ids,dtype=np.int64).tobytes()).hexdigest())
            for name,split in zip(('train','validation','test'),splits)}
        for seed in SEEDS:
            fitted={};reports=[]
            for weight in MMD_WEIGHTS:
                for reconstruction in RECONSTRUCTION_WEIGHTS:
                    print(f'任务3 第14步：留出{region}，初始化{seed}，重建{reconstruction}，MMD{weight}',flush=True)
                    model,report=fit(splits[0],splits[1],seed,weight,reconstruction,SETTINGS)
                    reports.append(report);fitted[(weight,reconstruction)]=model
                    context=dict(held_out=region,seed=seed,mmd_weight=weight,reconstruction_weight=reconstruction)
                    folder=args.output/region/str(seed)/f'mmd_{weight:g}'/f'reconstruction_{reconstruction:g}'
                    folder.mkdir(parents=True)
                    base.dump(folder/'training.json',report);base.dump(folder/'split_manifest.json',manifest)
                    np.savez(folder/'preprocessing.npz',median=prep.imputer.statistics_,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
                    torch.save(dict(state_dict=model.state_dict(),settings=dict(SETTINGS,reconstruction_weight=reconstruction),
                        features=list(base.FEATURES),route='prediction_only',seed=seed,mmd_weight=weight,
                        best_epoch=report['best_epoch'],selection_start_epoch=1),folder/'model.pt')
                    selection.append(dict(**context,best_epoch=report['best_epoch'],source_validation_auc=report['best_source_auc'],
                        trained_epochs=report['trained_epochs'],selected_state_sha256=report['selected_state_sha256']))
                    curves.extend(dict(**context,**r) for r in report['history'])
                    gradients.extend(dict(**context,**r) for r in report['gradient_diagnostics'])
                    if model_hash(model)!=report['selected_state_sha256']:raise RuntimeError('保存模型状态与选中状态不符')
                    runs+=1
            checks.append(dict(held_out=region,seed=seed,**control_checks(reports,SETTINGS)))
            # 任务3 第14步：同种子四条件均完成来源选模后，才评估留出地区。
            for (weight,reconstruction),model in fitted.items():
                context=dict(held_out=region,seed=seed,mmd_weight=weight,reconstruction_weight=reconstruction)
                for stage,part,split in zip(('source_train','source_validation','test'),frames,splits):
                    probabilities=predict(model,split.x,'prediction_only')
                    values=audit.evaluate(context,stage,part,probabilities)
                    if stage=='source_validation':
                        actual=next(r['auc'] for r in values if r['group']=='REGION_ALL' and r['subgroup']=='all')
                        expected=next(r['best_source_auc'] for r in reports if r['mmd_weight']==weight and r['reconstruction_weight']==reconstruction)
                        if abs(actual-expected)>1e-12:raise RuntimeError('评估模型与来源选模记录不一致')
                    rows.extend(values)
                    if stage!='source_train':
                        values,contribution=pathway.diagnostics(model,part,split,'prediction_only',context,stage)
                        rep.extend(values);path_use.append(contribution)
            for name,values in dict(metrics=rows,source_only_selection=selection,control_checks=checks,training_curves=curves,
                gradient_diagnostics=gradients,representation_diagnostics=rep,pathway_diagnostics=path_use).items():
                pd.DataFrame(values).to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
            metadata['completed_training_runs']=runs;base.dump(args.output/'experiment_metadata.json',metadata)
    table=pd.DataFrame(rows);macro=macro_table(table)
    delta=paired_differences(table,['held_out','seed','mmd_weight','stage','group','subgroup'])
    macro_delta=paired_differences(macro,['held_out','seed','mmd_weight','subgroup'])
    summary=delta[delta.stage=='test'].groupby(['held_out','mmd_weight','group','subgroup'],as_index=False).agg(
        mean_auc_change=('auc_change','mean'),min_auc_change=('auc_change','min'),max_auc_change=('auc_change','max'),
        positive_seed_count=('auc_change',lambda x:int((x>0).sum())),valid_seed_count=('auc_change','count'),
        mean_auprc_change=('auprc_change','mean'),mean_brier_change=('brier_change','mean'))
    for name,values in dict(region_and_global_macro=macro,matched_seed_differences=delta,
        region_macro_reconstruction_changes=macro_delta,seed_stability_summary=summary).items():
        values.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    if runs!=metadata['expected_training_runs']:raise RuntimeError('训练数量不完整')
    draw(macro,delta,pd.DataFrame(curves),pd.DataFrame(gradients),args.output)
    metadata['completed']=True;base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第14步完成：训练{runs}次，输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
