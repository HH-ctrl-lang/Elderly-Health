"""任务3 第15步：复核历史完整V5基线及分阶段训练表现。

逐折读取历史配置，保持损失、调度、学习率策略、来源划分和晚期选模。
早期快照仅作过程诊断；留出地区在训练和全部来源选择结束后才评估。
本步不进行损失消融，不把训练阶段的相关变化解释为某项损失的因果作用。
"""
import argparse
import copy
import hashlib
import inspect
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sparse_invariance_feature_pilot as base
import two_block_information_audit as audit
from elderly_health import training,models,losses,config as config_module
from elderly_health.config import TrainConfig
from elderly_health.data import prepare_splits

REGIONS=('east','central','west')
SEEDS=(42,43,44)
STAGES=('prediction_phase','sparse_phase','invariance_ramp','full_objective')
LOSS_NAMES=('prediction','zc_prediction','sparse','entropy','graph','irm','environment','independence')


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def full_start(cfg):
    """任务3 第15步：训练内部轮次从0计；完整权重对应的一基轮次。"""
    return max(cfg.sparse_start,cfg.irm_start)+cfg.ramp_epochs+1


def stage_at(epoch,cfg):
    if epoch>=full_start(cfg):return 'full_objective'
    if training.ramp(epoch-1,cfg.irm_start,cfg.ramp_epochs)>0:return 'invariance_ramp'
    if training.ramp(epoch-1,cfg.sparse_start,cfg.ramp_epochs)>0:return 'sparse_phase'
    return 'prediction_phase'


def loss_schedule(cfg):
    """任务3 第15步：零权重也明确记录，避免把已计算但未加权的项当成启用。"""
    rows=[]
    for epoch in range(1,cfg.epochs+1):
        sparse=training.ramp(epoch-1,cfg.sparse_start,cfg.ramp_epochs)
        invariant=training.ramp(epoch-1,cfg.irm_start,cfg.ramp_epochs)
        weights=dict(prediction=1.,zc_prediction=1.,sparse=sparse*cfg.lambda_sparse,
            entropy=sparse*cfg.lambda_entropy,graph=sparse*cfg.lambda_graph,
            irm=invariant*cfg.lambda_irm,environment=invariant*cfg.lambda_env,
            independence=invariant*cfg.lambda_independence)
        for name,weight in weights.items():
            rows.append(dict(epoch=epoch,stage=stage_at(epoch,cfg),loss=name,effective_weight=weight,
                enabled=weight>0,eligible_for_final_selection=epoch>=cfg.first_selection_epoch()))
    return pd.DataFrame(rows)


def validate_config(payload):
    """任务3 第15步：历史文件为唯一配置来源；不回退到默认值，不补开原本关闭的项。"""
    if payload['input_sha256']!=base.EXPECTED_HASH or payload['feature_names']!=list(base.FEATURES):
        raise ValueError('历史数据版本或特征顺序不一致')
    cfg=TrainConfig(**payload['config'])
    if cfg.model!='v5' or cfg.environment_granularity!='region' or cfg.seed!=42:
        raise ValueError('必须使用历史地区留出V5种子42配置')
    if cfg.irm_mode=='none' or cfg.independence=='none' or cfg.sparsity=='none':
        raise ValueError('历史完整配置的约束模式不完整')
    if min(cfg.lambda_sparse,cfg.lambda_irm,cfg.lambda_env,cfg.lambda_independence)<=0:
        raise ValueError('预期保留的约束权重必须为正')
    if cfg.lambda_graph!=0 or payload.get('graph_path') is not None:
        raise ValueError('当前历史基线没有图先验；遇到其他先验配置须单独核对')
    if cfg.first_selection_epoch()<full_start(cfg):raise ValueError('历史选模早于完整权重阶段')
    if cfg.device!='cpu':raise ValueError('本步只复核历史CPU基线')
    return cfg


class SourceSnapshots:
    """任务3 第15步：回调只复制参数和来源记录；不推理、不读目标、不消耗随机数。"""
    def __init__(self,cfg):
        self.cfg=cfg;self.best={};self.fixed={}
        boundaries={1,cfg.sparse_start+1,cfg.sparse_start+2,cfg.sparse_start+cfg.ramp_epochs+1,
                    cfg.irm_start+1,cfg.irm_start+2,full_start(cfg)-1,full_start(cfg)}
        self.boundaries={e for e in boundaries if 1<=e<=cfg.epochs}

    def __call__(self,model,record):
        epoch=record['epoch'];stage=stage_at(epoch,self.cfg)
        expected_sparse=training.ramp(epoch-1,self.cfg.sparse_start,self.cfg.ramp_epochs)
        expected_invariant=training.ramp(epoch-1,self.cfg.irm_start,self.cfg.ramp_epochs)
        if record['sparse_ramp']!=expected_sparse or record['invariant_ramp']!=expected_invariant:
            raise RuntimeError('实际损失排程与配置不一致')
        if record['eligible_for_selection']!=(epoch>=self.cfg.first_selection_epoch()):
            raise RuntimeError('实际最终选模窗口与配置不一致')
        required=['train_prediction','train_zc_prediction']
        if expected_sparse>0:required+=['train_sparse','train_entropy']
        if expected_invariant>0:required+=['train_irm','train_environment','train_independence']
        if any(k not in record for k in required):raise RuntimeError('完整训练缺少预期损失记录')
        weights=dict(prediction=1.,zc_prediction=1.,sparse=expected_sparse*self.cfg.lambda_sparse,
            entropy=expected_sparse*self.cfg.lambda_entropy,graph=expected_sparse*self.cfg.lambda_graph,
            irm=expected_invariant*self.cfg.lambda_irm,environment=expected_invariant*self.cfg.lambda_env,
            independence=expected_invariant*self.cfg.lambda_independence)
        reconstructed=sum(weight*record.get('train_'+name,0.) for name,weight in weights.items())
        if abs(reconstructed-record['train_total'])>2e-6:
            raise RuntimeError('逐项加权损失与训练总损失不一致')
        better=stage not in self.best or record['val_auc']>self.best[stage]['record']['val_auc']
        if better or epoch in self.boundaries:
            snapshot=dict(record=copy.deepcopy(record),state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if better:self.best[stage]=snapshot
            if epoch in self.boundaries:self.fixed[epoch]=snapshot


def verify_history(actual,expected):
    """任务3 第15步：种子42先核对来源训练轨迹，不依据目标结果决定是否重现。"""
    if actual['best_epoch']!=expected['best_epoch'] or len(actual['history'])!=len(expected['history']):
        raise RuntimeError('历史选中轮次或停止轮次未复现')
    maximum=0.
    for a,b in zip(actual['history'],expected['history']):
        if a['epoch']!=b['epoch'] or a['eligible_for_selection']!=b['eligible_for_selection']:
            raise RuntimeError('历史轮次或选模资格不一致')
        for key in ('val_auc','val_loss','sparse_ramp','invariant_ramp'):
            difference=abs(a[key]-b[key]);maximum=max(maximum,difference)
            if difference>2e-6:raise RuntimeError(f'历史来源曲线未复现：轮次{a["epoch"]}，字段{key}')
    return dict(source_history_reproduced=True,max_source_history_difference=maximum,
                selected_epoch=actual['best_epoch'],trained_epochs=len(actual['history']))


def macro_table(table):
    sites=table[(table.stage=='test')&(table.group!='REGION_ALL')]
    keys=['seed','checkpoint','subgroup']
    region=sites.groupby(['held_out']+keys,as_index=False)[['auc','auprc','brier']].mean()
    global_=sites.groupby(keys,as_index=False)[['auc','auprc','brier']].mean();global_['held_out']='ALL'
    return pd.concat([region,global_],ignore_index=True)


def draw(schedule,curve,macro,boundary,output):
    """任务3 第15步：明确区分完整基线与训练过程诊断，不展示目标最优轮次。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5))
    values=schedule.pivot(index='epoch',columns='loss',values='effective_weight')
    for name in LOSS_NAMES:ax.plot(values.index,values[name],label=name)
    first=int(schedule.loc[schedule.eligible_for_final_selection,'epoch'].min())
    ax.axvline(first,color='black',ls='--',label='Final selection starts')
    ax.set_xlabel('Training epoch');ax.set_ylabel('Effective coefficient (not gradient strength)')
    ax.set_title('Historical full V5 objective: zero coefficients stay zero');ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(output/'figures/01_full_v5_loss_schedule.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        for seed in SEEDS:
            part=curve[(curve.held_out==region)&(curve.seed==seed)]
            ax.plot(part.epoch,part.val_auc,label=f'Seed {seed}')
        ax.axvline(first,color='black',ls='--');ax.set_title(region)
        ax.set_xlabel('Training epoch');ax.set_ylabel('Source validation AUC');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'figures/02_full_v5_source_learning.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    stages=['stage_best_'+s for s in STAGES]
    labels=['Prediction phase','Sparse phase','Invariance ramp','Full-objective phase']
    for ax,metric in zip(axes,('auc','auprc','brier')):
        for region in REGIONS:
            part=macro[(macro.held_out==region)&(macro.subgroup=='all')]
            values=part.groupby('checkpoint')[metric].mean().reindex(stages)
            ax.plot(range(len(stages)),values,'o-',label=region)
        ax.set_xticks(range(len(stages)));ax.set_xticklabels(labels,rotation=20,ha='right',fontsize=8)
        ax.set_title(metric.upper()+(' (lower better)' if metric=='brier' else ' (higher better)'))
        ax.set_ylabel('Equal-site macro');ax.legend(fontsize=8)
    fig.suptitle('Source-best within each phase; early phases are diagnostics, not final models')
    fig.tight_layout();fig.savefig(output/'figures/03_phase_diagnostics_and_full_baseline.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=boundary[(boundary.held_out==region)&(boundary.subgroup=='all')&(boundary.group!='REGION_ALL')]
        for site,values in part.groupby('group'):
            mean=values.groupby('epoch').auc.mean();ax.plot(mean.index,mean,'o-',label=str(site))
        ax.axvline(first,color='black',ls='--');ax.set_title(region)
        ax.set_xlabel('Fixed schedule boundary epoch');ax.set_ylabel('Held-out site AUC');ax.legend(fontsize=7)
    fig.suptitle('Post-training diagnostics only; no target-based checkpoint selection')
    fig.tight_layout();fig.savefig(output/'figures/04_site_schedule_boundary_diagnostics.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=base.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=base.ROOT/'outputs/analysis_2611/task3_step15_full_v5_training_stage_audit')
    args=parser.parse_args()
    if audit.digest(args.data)!=base.EXPECTED_HASH:raise ValueError('固定CSV版本不一致')
    if 'epoch_callback' not in inspect.signature(training.fit).parameters:raise RuntimeError('请先拉取支持只读快照回调的当前训练代码')
    payloads={r:json.loads((args.previous/r/'v5/config.json').read_text(encoding='utf-8')) for r in REGIONS}
    configs={r:validate_config(p) for r,p in payloads.items()}
    if len({json.dumps(c.to_dict(),sort_keys=True) for c in configs.values()})!=1:raise ValueError('三折历史配置不一致，须先核对')
    frame=base.attach_regions(base.read_raw(args.data));args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'figures').mkdir()
    metadata=dict(completed=False,model='historical_full_v5',seeds=SEEDS,split_seed=42,expected_training_runs=9,
        input_sha256=base.EXPECTED_HASH,config=configs['east'].to_dict(),
        full_weight_start_epoch=full_start(configs['east']),early_checkpoints_are_diagnostics_only=True,
        target_used_for_training_or_selection=False,individual_records_exported=False,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        runner_sha256=audit.digest(Path(__file__)),
        code_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (training,models,losses,config_module)},
        limitations=['完整指历史V5配置；熵惩罚与图先验原权重为0，不擅自补开。',
            'v11为项目梯度方差启发式，不等同于标准IRMv1；门控值不等于因果重要性。',
            '阶段同时涉及训练时间、学习率调度和多项损失变化，不能归因于某一损失。',
            '早期阶段最优只依据来源验证选择，不能当作完整目标下的最终模型或按目标表现反向选阶段。',
            '固定边界诊断在训练结束后评估；目标已多次探索，三初始化不是独立人群。'])
    base.dump(args.output/'experiment_metadata.json',metadata)
    schedule=loss_schedule(configs['east']);schedule.to_csv(args.output/'loss_schedule.csv',index=False,encoding='utf-8-sig')
    rows=[];boundaries=[];selections=[];curves=[];checks=[];runs=0
    for region in REGIONS:
        cfg0=configs[region];frames=base.held_out_split(frame,region,seed=42,column=base.REGION)
        splits,prep=prepare_splits(frames,environment_column=base.REGION)
        if any(set(a.sample_ids)&set(b.sample_ids) for i,a in enumerate(splits) for b in splits[i+1:]):raise RuntimeError('人员划分重叠')
        expected=json.loads((args.previous/region/'v5/training.json').read_text(encoding='utf-8'))
        for seed in SEEDS:
            cfg=replace(cfg0,seed=seed);capture=SourceSnapshots(cfg)
            print(f'任务3 第15步：留出{region}，种子{seed}，保留历史完整目标，第{cfg.first_selection_epoch()}轮起选模',flush=True)
            model,report=training.fit(splits[0],splits[1],cfg,epoch_callback=capture)
            if set(capture.best)!=set(STAGES):raise RuntimeError('训练未覆盖全部预定阶段')
            stage_full=capture.best['full_objective']
            if stage_full['record']['epoch']!=report['best_epoch'] or state_hash(stage_full['state'])!=state_hash(model.state_dict()):
                raise RuntimeError('完整阶段的来源最优与正式选中模型不一致')
            reproduction=verify_history(report,expected) if seed==42 else dict(source_history_reproduced=None)
            checks.append(dict(held_out=region,seed=seed,all_expected_losses_recorded=True,loss_schedule_verified=True,
                full_stage_selection_verified=True,**reproduction))
            folder=args.output/region/str(seed);folder.mkdir(parents=True)
            base.dump(folder/'config.json',cfg.to_dict());base.dump(folder/'training.json',report)
            torch.save(dict(state_dict=model.state_dict(),config=cfg.to_dict(),features=list(base.FEATURES),
                best_epoch=report['best_epoch'],env_map=report['env_map']),folder/'full_model.pt')
            np.savez(folder/'preprocessing.npz',median=prep.imputer.statistics_,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
            curves.extend(dict(held_out=region,seed=seed,phase=stage_at(r['epoch'],cfg),
                **{k:v for k,v in r.items() if k!='mask'}) for r in report['history'])
            # 任务3 第15步：全部快照已按来源记录确定，此后才使用目标数据进行诊断。
            chosen={'stage_best_'+stage:shot for stage,shot in capture.best.items()}
            chosen['final_full_model']=stage_full
            for name,shot in chosen.items():
                model.load_state_dict(shot['state']);model.eval()
                selections.append(dict(held_out=region,seed=seed,checkpoint=name,epoch=shot['record']['epoch'],
                    source_validation_auc=shot['record']['val_auc'],final_model=name=='final_full_model',
                    state_sha256=state_hash(shot['state'])))
                context=dict(held_out=region,seed=seed,checkpoint=name,epoch=shot['record']['epoch'])
                for stage,part,split in zip(('source_train','source_validation','test'),frames,splits):
                    p=training.predict(model,split.x,torch.device('cpu'))
                    rows.extend(audit.evaluate(context,stage,part,p))
                if name!='final_full_model':torch.save(dict(state_dict=shot['state'],epoch=shot['record']['epoch'],diagnostic_only=True),folder/(name+'.pt'))
            for epoch,shot in capture.fixed.items():
                model.load_state_dict(shot['state']);model.eval()
                p=training.predict(model,splits[2].x,torch.device('cpu'))
                boundaries.extend(audit.evaluate(dict(held_out=region,seed=seed,epoch=epoch), 'test',frames[2],p))
            runs+=1
            for name,values in dict(metrics=rows,boundary_metrics=boundaries,source_only_checkpoints=selections,
                training_curves=curves,control_checks=checks).items():pd.DataFrame(values).to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
            metadata['completed_training_runs']=runs;base.dump(args.output/'experiment_metadata.json',metadata)
    table=pd.DataFrame(rows);macro=macro_table(table);macro.to_csv(args.output/'region_and_global_macro.csv',index=False,encoding='utf-8-sig')
    draw(schedule,pd.DataFrame(curves),macro,pd.DataFrame(boundaries),args.output)
    if runs!=metadata['expected_training_runs']:raise RuntimeError('训练数量不完整')
    metadata['completed']=True;base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第15步完成：复核完整V5，训练{runs}次，输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
