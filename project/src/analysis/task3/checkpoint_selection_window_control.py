"""任务3 第13步：同一训练轨迹下，控制来源验证选模的起始轮次。

只用预测块分类，固定MMD权重0/0.1。比较第1轮起可选与第91轮起可选；
两窗口共享全部优化过程，沿用晚期窗口的早停规则，留出地区不参与选模。
本步检查训练政策，不是因果识别，也不据目标表现挑选窗口或参数。
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
SETTINGS=pathway.SETTINGS.copy()
WINDOWS=('from_epoch1','from_epoch91')
LABELS={'from_epoch1':'Selectable from epoch 1','from_epoch91':'Selectable from epoch 91'}


class CheckpointTracker:
    """任务3 第13步：严格大于才更新；平分保留最早轮次，早期窗口不改变停训。"""
    def __init__(self,late_start):
        self.late_start=late_start
        self.best={w:None for w in WINDOWS}
        self.stale=0

    def observe(self,epoch,auc,state):
        if not np.isfinite(auc):raise ValueError('来源验证AUC不是有限值')
        for window in WINDOWS:
            if window=='from_epoch91' and epoch<self.late_start:continue
            old=self.best[window]
            if old is None or auc>old['auc']:
                self.best[window]=dict(epoch=epoch,auc=float(auc),state=copy.deepcopy(state))
                if window=='from_epoch91':self.stale=0
            elif window=='from_epoch91':self.stale+=1


def fit(train,val,seed,mmd_weight,settings):
    """任务3 第13步：同一轨迹记录两套来源选模；晚期规则独立决定何时停训。"""
    route="prediction_only"
    if not 1 <= settings["selection_start_epoch"] <= settings["epochs"]:
        raise ValueError("选模起始轮次超出训练范围")
    tracker=CheckpointTracker(settings["selection_start_epoch"])
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
    history=[]
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
        tracker.observe(epoch,source_auc,model.state_dict())
        if epoch==1 or epoch%30==0:
            print(f'任务3 第13步：种子{seed}，路径{route}，约束{mmd_weight}，轮次{epoch}，来源验证AUC={source_auc:.4f}',flush=True)
        if eligible and tracker.stale>=settings['patience']:break
    models={}
    for window,checkpoint in tracker.best.items():
        if checkpoint is None:raise RuntimeError('没有可选来源检查点')
        selected=build_model(seed,train.x.shape[1])
        selected.load_state_dict(checkpoint['state']);selected.eval()
        models[window]=selected
    report=dict(seed=seed,route=route,mmd_weight=mmd_weight,
        initial_state_sha256=initial,initial_probability_sha256=initial_probability_hash,
        sampling_prefix_sha256=prefixes,history=history,trained_epochs=len(history),
        parameter_count=sum(p.numel() for p in model.parameters()),
        selections={w:dict(best_epoch=c['epoch'],best_source_auc=c['auc'],
                          selected_state_sha256=pathway.model_hash(models[w]))
                    for w,c in tracker.best.items()})
    return models,report


def selection_checks(report,models,settings):
    """任务3 第13步：独立从日志重算两个最优轮次，检查模型和来源选择一致。"""
    history=report['history'];selections=report['selections']
    for window in WINDOWS:
        start=1 if window=='from_epoch1' else settings['selection_start_epoch']
        candidates=[r for r in history if r['epoch']>=start]
        expected=max(candidates,key=lambda r:r['source_validation_auc'])
        selected=selections[window]
        if (expected['epoch']!=selected['best_epoch'] or
            expected['source_validation_auc']!=selected['best_source_auc'] or
            model_hash(models[window])!=selected['selected_state_sha256']):
            raise RuntimeError('检查点选择或保存状态不匹配')
    early,late=(selections[w] for w in WINDOWS)
    if early['best_source_auc']<late['best_source_auc']:raise RuntimeError('嵌套窗口的来源选择不符合预期')
    same=early['best_epoch']==late['best_epoch']
    if same and early['selected_state_sha256']!=late['selected_state_sha256']:
        raise RuntimeError('同轮次的检查点参数不一致')
    return dict(source_argmax_verified=True,checkpoint_hash_verified=True,
        same_training_trajectory=True,same_selected_epoch=same,
        early_selected_before_late_start=early['best_epoch']<settings['selection_start_epoch'])


def window_differences(table,keys):
    """任务3 第13步：两窗口同种子、同人群配对，报告全部指标变化。"""
    early=table[table.window==WINDOWS[0]];late=table[table.window==WINDOWS[1]]
    paired=early.merge(late,on=keys,suffixes=('_early','_late'),validate='one_to_one')
    if len(paired)!=len(early) or len(paired)!=len(late):raise RuntimeError('窗口配对缺行')
    result=paired[keys].copy()
    for metric in ('auc','auprc','brier'):
        result[metric+'_early']=paired[metric+'_early']
        result[metric+'_late']=paired[metric+'_late']
        result[metric+'_change']=paired[metric+'_early']-paired[metric+'_late']
    return result


def macro_table(table):
    sites=table[(table.stage=='test')&(table.group!='REGION_ALL')]
    keys=['seed','mmd_weight','window','subgroup']
    region=sites.groupby(['held_out']+keys,as_index=False)[['auc','auprc','brier']].mean()
    global_=sites.groupby(keys,as_index=False)[['auc','auprc','brier']].mean();global_['held_out']='ALL'
    return pd.concat([region,global_],ignore_index=True)


def draw(macro,delta,selection,curves,output):
    """任务3 第13步：来源学习曲线、地区指标、站点变化及来源到目标的对应。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    colors={0.:'tab:blue',.1:'tab:orange'}
    for ax,region in zip(axes,REGIONS):
        for weight in MMD_WEIGHTS:
            part=curves[(curves.held_out==region)&(curves.mmd_weight==weight)]
            for seed in SEEDS:
                one=part[part.seed==seed]
                ax.plot(one.epoch,one.source_validation_auc,color=colors[weight],alpha=.4,
                        label=f'MMD={weight:g}' if seed==SEEDS[0] else None)
            selected=selection[(selection.held_out==region)&(selection.mmd_weight==weight)]
            for window,marker in zip(WINDOWS,('o','x')):
                one=selected[selected.window==window]
                ax.scatter(one.best_epoch,one.source_validation_auc,marker=marker,color=colors[weight],s=40)
        ax.axvline(SETTINGS['selection_start_epoch'],color='black',ls='--',lw=1)
        ax.set_title(region);ax.set_xlabel('Training epoch');ax.legend(fontsize=8)
    axes[0].set_ylabel('Source validation AUC')
    fig.suptitle('Same trajectory: circle = from epoch 1; cross = from epoch 91')
    fig.tight_layout();fig.savefig(output/'figures/01_source_learning_and_selection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,weight in enumerate(MMD_WEIGHTS):
        for ax,metric in zip(axes[row],('auc','auprc','brier')):
            part=macro[(macro.mmd_weight==weight)&(macro.subgroup=='all')]
            values=part.groupby(['held_out','window'])[metric].mean().unstack().reindex([*REGIONS,'ALL'])
            values=values.reindex(columns=WINDOWS).rename(columns=LABELS)
            values.plot.bar(ax=ax,rot=0)
            ax.set_title(f'MMD={weight:g}: {metric.upper()} '+('(lower better)' if metric=='brier' else '(higher better)'))
            ax.set_ylabel('Equal-site macro');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/02_region_checkpoint_tradeoffs.png',dpi=180);plt.close(fig)
    site=delta[(delta.stage=='test')&(delta.subgroup=='all')&(delta.group!='REGION_ALL')]
    order=[(r,str(s)) for r in REGIONS for s in sorted(site.loc[site.held_out==r,'group'].unique())]
    matrices=[]
    for weight in MMD_WEIGHTS:
        part=site[site.mmd_weight==weight].pivot(index=['held_out','group'],columns='seed',values='auc_change')
        part=part.reindex(index=pd.MultiIndex.from_tuples(order),columns=SEEDS)
        part['Mean']=part.mean(axis=1);matrices.append(part)
    limit=max(.01,max(float(np.nanmax(np.abs(m.values))) for m in matrices))
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True)
    for ax,weight,values in zip(axes,MMD_WEIGHTS,matrices):
        im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(values.columns)));ax.set_xticklabels(values.columns)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([f'{r}: {s}' for r,s in order])
        ax.set_title(f'MMD={weight:g}: early-window minus late-window')
        for i in range(len(values)):
            for j in range(len(values.columns)):
                v=values.iloc[i,j];ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(v)>.65*limit else 'black')
    fig.subplots_adjust(left=.17,right=.86,wspace=.12,bottom=.08,top=.94)
    cax=fig.add_axes([.90,.18,.015,.65]);fig.colorbar(im,cax=cax,label='Paired site AUC change')
    fig.savefig(output/'figures/03_site_checkpoint_seed_changes.png',dpi=180);plt.close(fig)
    chosen=selection.pivot(index=['held_out','seed','mmd_weight'],columns='window',values='source_validation_auc')
    chosen['source_gain']=chosen[WINDOWS[0]]-chosen[WINDOWS[1]]
    target=window_differences(macro[macro.subgroup=='all'],['held_out','seed','mmd_weight','subgroup'])
    joined=target.merge(chosen[['source_gain']].reset_index(),on=['held_out','seed','mmd_weight'],validate='one_to_one')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,weight in zip(axes,MMD_WEIGHTS):
        for region in REGIONS:
            part=joined[(joined.mmd_weight==weight)&(joined.held_out==region)]
            ax.scatter(part.source_gain,part.auc_change,label=region)
            for _,r in part.iterrows():ax.annotate(str(r.seed),(r.source_gain,r.auc_change),fontsize=8,xytext=(3,3),textcoords='offset points')
        ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Source validation AUC gain');ax.set_title(f'MMD={weight:g}');ax.legend(fontsize=8)
    axes[0].set_ylabel('Held-out equal-site macro AUC change')
    fig.tight_layout();fig.savefig(output/'figures/04_source_gain_vs_target_gain.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--output',type=Path,default=base.ROOT/'outputs/analysis_2611/task3_step13_checkpoint_selection_window_control')
    args=parser.parse_args()
    if audit.digest(args.data)!=base.EXPECTED_HASH:raise ValueError('固定数据版本不一致')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame=base.attach_regions(base.read_raw(args.data))
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    metadata=dict(completed=False,settings=SETTINGS,split_seed=42,seeds=SEEDS,mmd_weights=MMD_WEIGHTS,
        route='prediction_only',windows={WINDOWS[0]:1,WINDOWS[1]:SETTINGS['selection_start_epoch']},
        stopping_rule='late_window_patience_only',input_sha256=base.EXPECTED_HASH,
        expected_training_runs=len(REGIONS)*len(SEEDS)*len(MMD_WEIGHTS),
        expected_selected_models=len(REGIONS)*len(SEEDS)*len(MMD_WEIGHTS)*len(WINDOWS),
        target_used_for_training_or_selection=False,individual_records_exported=False,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        runner_sha256=audit.digest(Path(__file__)),
        code_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (pathway,base,audit,adapter)},
        limitations=['只检验选模起始轮次，不能据此认定因果机制或所有解耦方法均有效或无效。',
            '从第1轮起可选的窗口包含晚期模型，不等于强迫早停；两窗口使用同一轨迹。',
            '沿用来源合并验证AUC、数值编码、重建损失和晚期早停规则，不同时改变其他因素。',
            '早期窗口搜索候选更多，来源验证收益可能包含选择噪声；不能据此保证目标收益。',
            '来源验证重复使用，目标地区已多次探索；三初始化不是独立人群，不作确认性显著性结论。'])
    base.dump(args.output/'experiment_metadata.json',metadata)
    rows=[];selection=[];checks=[];curves=[];rep=[];path_use=[];runs=0
    for region in REGIONS:
        frames=base.held_out_split(frame,region,seed=42,column=base.REGION)
        splits,prep=prepare_splits(frames,environment_column=base.REGION)
        if any(set(a.sample_ids)&set(b.sample_ids) for i,a in enumerate(splits) for b in splits[i+1:]):raise RuntimeError('人员划分重叠')
        for seed in SEEDS:
            fitted={};pair_reports=[]
            for weight in MMD_WEIGHTS:
                print(f'任务3 第13步：留出{region}，初始化{seed}，约束{weight}，同一轨迹选择两个模型',flush=True)
                models,report=fit(splits[0],splits[1],seed,weight,SETTINGS)
                checks.append(dict(held_out=region,seed=seed,mmd_weight=weight,**selection_checks(report,models,SETTINGS)))
                pair_reports.append(report)
                folder=args.output/region/str(seed)/f'mmd_{weight:g}';folder.mkdir(parents=True)
                base.dump(folder/'training.json',report)
                np.savez(folder/'preprocessing.npz',median=prep.imputer.statistics_,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
                curves.extend(dict(held_out=region,seed=seed,mmd_weight=weight,**r) for r in report['history'])
                for window,model in models.items():
                    context=dict(held_out=region,seed=seed,mmd_weight=weight,window=window)
                    fitted[(weight,window)]=model
                    chosen=report['selections'][window]
                    selection.append(dict(**context,best_epoch=chosen['best_epoch'],source_validation_auc=chosen['best_source_auc'],
                                          trained_epochs=report['trained_epochs'],selected_state_sha256=chosen['selected_state_sha256']))
                    torch.save(dict(state_dict=model.state_dict(),settings=SETTINGS,features=list(base.FEATURES),route='prediction_only',
                        seed=seed,mmd_weight=weight,window=window,best_epoch=chosen['best_epoch']),folder/(window+'_model.pt'))
                runs+=1
            # 任务3 第13步：MMD组间也核对初值与共同轮次抽样；两窗口没有各自独立训练。
            for key in ('initial_state_sha256','initial_probability_sha256','parameter_count'):
                if len({r[key] for r in pair_reports})!=1:raise RuntimeError('约束组间初始条件不一致')
            common=set.intersection(*(set(r['sampling_prefix_sha256']) for r in pair_reports))
            if not common or any(len({r['sampling_prefix_sha256'][e] for r in pair_reports})!=1 for e in common):
                raise RuntimeError('约束组共同轮次抽样不一致')
            for check in checks[-len(MMD_WEIGHTS):]:
                check.update(initial_conditions_matched=True,sampling_matched=True,common_sampling_epochs=len(common))
            # 任务3 第13步：本种子所有来源选模结束后，才使用留出地区计算聚合指标。
            for (weight,window),model in fitted.items():
                context=dict(held_out=region,seed=seed,mmd_weight=weight,window=window)
                for stage,part,split in zip(('source_train','source_validation','test'),frames,splits):
                    rows.extend(audit.evaluate(context,stage,part,predict(model,split.x,'prediction_only')))
                    if stage!='source_train':
                        values,contribution=pathway.diagnostics(model,part,split,'prediction_only',context,stage)
                        rep.extend(values);path_use.append(contribution)
            for name,values in dict(metrics=rows,source_only_selection=selection,control_checks=checks,training_curves=curves,
                                    representation_diagnostics=rep,pathway_diagnostics=path_use).items():
                pd.DataFrame(values).to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
            metadata['completed_training_runs']=runs;metadata['completed_selected_models']=len(selection)
            base.dump(args.output/'experiment_metadata.json',metadata)
    table=pd.DataFrame(rows);macro=macro_table(table)
    delta=window_differences(table,['held_out','seed','mmd_weight','stage','group','subgroup'])
    macro_delta=window_differences(macro,['held_out','seed','mmd_weight','subgroup'])
    summary=delta[delta.stage=='test'].groupby(['held_out','mmd_weight','group','subgroup'],as_index=False).agg(
        mean_auc_change=('auc_change','mean'),min_auc_change=('auc_change','min'),max_auc_change=('auc_change','max'),
        positive_seed_count=('auc_change',lambda x:int((x>0).sum())),valid_seed_count=('auc_change','count'),
        mean_auprc_change=('auprc_change','mean'),mean_brier_change=('brier_change','mean'))
    for name,values in dict(region_and_global_macro=macro,matched_seed_differences=delta,
                            region_macro_window_changes=macro_delta,seed_stability_summary=summary).items():
        values.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    if runs!=metadata['expected_training_runs'] or len(selection)!=metadata['expected_selected_models']:
        raise RuntimeError('实验运行数量不完整')
    draw(macro,delta,pd.DataFrame(selection),pd.DataFrame(curves),args.output)
    metadata['completed']=True;base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第13步完成：训练{runs}次，保存{len(selection)}个选中模型，输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
