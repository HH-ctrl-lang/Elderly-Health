"""任务3 第15步：只读审计已有完整V5配置、训练日志和最终聚合指标。

默认零训练、零模型加载，不调用训练器，不要求重新产生相同的停止轮次。
早期阶段仅汇总来源验证日志；没有保存的早期目标指标明确标为不可用。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
REGIONS=('east','central','west')
LOSSES=('prediction','zc_prediction','sparse','entropy','graph','irm','environment','independence')
STAGES=('prediction_phase','sparse_phase','invariance_ramp','full_objective')
EXPECTED_HASH='969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de'
AUDIT_MODE='existing_artifacts_only'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def ramp(epoch,start,span):
    """任务3 第15步：与原训练器一致，输入epoch为一基轮次。"""
    return max(0.,min(1.,(epoch-1-start)/span))


def full_start(cfg):
    return max(cfg['sparse_start'],cfg['irm_start'])+cfg['ramp_epochs']+1


def first_selection(cfg):
    explicit=cfg.get('selection_start_epoch')
    return explicit if explicit is not None else cfg['irm_start']+1


def stage_at(epoch,cfg):
    if epoch>=full_start(cfg):return 'full_objective'
    if ramp(epoch,cfg['irm_start'],cfg['ramp_epochs'])>0:return 'invariance_ramp'
    if ramp(epoch,cfg['sparse_start'],cfg['ramp_epochs'])>0:return 'sparse_phase'
    return 'prediction_phase'


def weights_at(epoch,cfg):
    """任务3 第15步：计算实际有效系数，保留历史零权重设置。"""
    s=ramp(epoch,cfg['sparse_start'],cfg['ramp_epochs'])
    i=ramp(epoch,cfg['irm_start'],cfg['ramp_epochs'])
    return dict(prediction=1.,zc_prediction=1.,sparse=s*cfg['lambda_sparse'],
        entropy=s*cfg['lambda_entropy'],graph=s*cfg['lambda_graph'],irm=i*cfg['lambda_irm'],
        environment=i*cfg['lambda_env'],independence=i*cfg['lambda_independence'])


def audit_fold(region,payload,report,metrics):
    """任务3 第15步：审计历史文件自身的一致性，发现差异只记录，不触发重训。"""
    cfg=payload['config']
    if cfg['model']!='v5' or cfg['environment_granularity']!='region':raise ValueError('历史文件不是地区环境完整V5')
    if payload['input_sha256']!=EXPECTED_HASH:raise ValueError('历史数据版本不一致')
    if cfg['ramp_epochs']<=0:raise ValueError('损失渐增跨度必须为正')
    if cfg['lambda_graph']!=0 or payload.get('graph_path') is not None:
        raise ValueError('本步针对无图先验的历史V5配置，其他配置须单独核对')
    if cfg['irm_mode']=='none' or cfg['independence']=='none' or cfg['sparsity']=='none':
        raise ValueError('预期完整基线的约束模式被关闭')
    history=report['history']
    if not history:raise ValueError('历史训练日志为空')
    schedule=[];curves=[];phases=[];checks=[]
    def check(name,passed,detail):
        checks.append(dict(held_out=region,check=name,passed=bool(passed),detail=detail))
    check('selection_after_full_weight',first_selection(cfg)>=full_start(cfg),'最终选择窗口不得早于完整权重阶段')
    check('report_selection_matches_config',report.get('selection_start_epoch')==first_selection(cfg),'历史配置与日志选模窗口一致')
    check('epochs_contiguous',[r['epoch'] for r in history]==list(range(1,len(history)+1)),'历史日志轮次连续')
    for record in history:
        epoch=record['epoch'];weights=weights_at(epoch,cfg);phase=stage_at(epoch,cfg)
        expected_s=ramp(epoch,cfg['sparse_start'],cfg['ramp_epochs'])
        expected_i=ramp(epoch,cfg['irm_start'],cfg['ramp_epochs'])
        check(f'schedule_epoch_{epoch}',abs(record['sparse_ramp']-expected_s)<1e-12 and
            abs(record['invariant_ramp']-expected_i)<1e-12 and
            record['eligible_for_selection']==(epoch>=first_selection(cfg)),'损失渐增与选模资格核对')
        row=dict(held_out=region,seed=cfg['seed'],epoch=epoch,phase=phase,
            val_auc=record['val_auc'],val_loss=record['val_loss'],train_total=record['train_total'],
            eligible_for_selection=record['eligible_for_selection'])
        missing=[];total=0.
        for loss,weight in weights.items():
            key='train_'+loss
            if weight>0 and key not in record:missing.append(loss)
            raw=record.get(key)
            contribution=weight*raw if raw is not None else 0. if weight==0 else None
            row['weighted_'+loss]=contribution
            if contribution is not None:total+=contribution
            schedule.append(dict(held_out=region,epoch=epoch,phase=phase,loss=loss,
                effective_weight=weight,raw_logged=raw is not None,enabled=weight>0,
                raw_training_loss=raw,weighted_training_contribution=contribution))
        difference=abs(total-record['train_total']) if not missing else None
        row['weighted_total_difference']=difference
        check(f'objective_epoch_{epoch}',not missing and difference<=2e-6,
              '缺少有效损失：'+('|'.join(missing) if missing else '无')+'；检查逐项加权总和')
        curves.append(row)
    eligible=[r for r in history if r['eligible_for_selection']]
    if not eligible:raise ValueError('历史日志没有可选完整模型')
    recomputed=max(eligible,key=lambda r:r['val_auc'])
    check('selected_epoch_matches_source_argmax',recomputed['epoch']==report['best_epoch'], '平分时保留最早轮次')
    check('selected_auc_matches_source_argmax',abs(recomputed['val_auc']-report['best_val_auc'])<=2e-6,'来源AUC重算，不训练')
    for phase in STAGES:
        part=[r for r in history if stage_at(r['epoch'],cfg)==phase]
        best=max(part,key=lambda r:r['val_auc']) if part else None
        phases.append(dict(held_out=region,seed=cfg['seed'],phase=phase,available=bool(part),
            start_epoch=part[0]['epoch'] if part else None,end_epoch=part[-1]['epoch'] if part else None,
            best_source_epoch=best['epoch'] if best else None,best_source_auc=best['val_auc'] if best else None,
            final_model_candidate=phase=='full_objective',target_phase_metrics_available=False))
    site=[]
    for group,values in metrics['per_site'].items():
        site.append(dict(held_out=region,seed=cfg['seed'],group=str(group),scope='site',
            selected_epoch=report['best_epoch'],**{k:values.get(k) for k in ('auc','auprc','brier','n','positive_rate')}))
    pooled=dict(held_out=region,seed=cfg['seed'],group='REGION_ALL',scope='region_pooled',
        selected_epoch=report['best_epoch'],**{k:metrics['main'].get(k) for k in ('auc','auprc','brier','n','positive_rate')})
    macro=dict(held_out=region,seed=cfg['seed'],group='SITE_MACRO',scope='equal_site_macro',selected_epoch=report['best_epoch'])
    for metric in ('auc','auprc','brier'):
        values=[r[metric] for r in site]
        macro[metric]=float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None
    return schedule,curves,phases,checks,site+[pooled,macro]


def draw(schedule,curves,final,output):
    """任务3 第15步：只展示已有信息，不生成不存在的早期目标曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=schedule[schedule.held_out==region]
        values=part.pivot(index='epoch',columns='loss',values='effective_weight')
        for loss in LOSSES:ax.plot(values.index,values[loss],label=loss)
        ax.set_title(region);ax.set_xlabel('Recorded epoch');ax.set_ylabel('Effective loss coefficient');ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Historical full V5: configured zero weights remain zero')
    fig.tight_layout();fig.savefig(output/'figures/01_existing_full_v5_loss_schedule.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,region in zip(axes,REGIONS):
        part=curves[curves.held_out==region]
        ax.plot(part.epoch,part.val_auc,label='Source validation AUC')
        selected=final[(final.held_out==region)&(final.scope=='region_pooled')].iloc[0].selected_epoch
        first=int(part.loc[part.eligible_for_selection,'epoch'].min())
        ax.axvline(first,color='grey',ls='--',label='Eligible selection starts')
        ax.axvline(selected,color='black',ls=':',label='Historical selected epoch')
        ax.set_title(region);ax.set_xlabel('Recorded epoch');ax.set_ylabel('Source validation AUC');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'figures/02_existing_source_learning.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,region in zip(axes,REGIONS):
        part=curves[curves.held_out==region]
        for loss in LOSSES:
            if schedule[(schedule.held_out==region)&(schedule.loss==loss)].enabled.any():
                ax.plot(part.epoch,part['weighted_'+loss],label=loss)
        ax.set_title(region);ax.set_xlabel('Recorded epoch');ax.set_ylabel('Weighted mean training contribution');ax.legend(fontsize=7)
    fig.suptitle('Training loss values are not gradient strength or causal importance')
    fig.tight_layout();fig.savefig(output/'figures/03_existing_weighted_loss_contributions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,metric in zip(axes,('auc','auprc','brier')):
        for region in REGIONS:
            part=final[(final.held_out==region)&(final.scope=='site')]
            ax.scatter([region]*len(part),part[metric],alpha=.7,label=region)
            macro=final[(final.held_out==region)&(final.scope=='equal_site_macro')].iloc[0][metric]
            ax.scatter([region],[macro],marker='_',s=350,color='black')
        ax.set_title(metric.upper()+(' (lower better)' if metric=='brier' else ' (higher better)'))
        ax.set_ylabel('Stored final-model performance');ax.legend(fontsize=7)
    fig.suptitle('Historical final checkpoint only: dots = sites; black bar = equal-site macro')
    fig.tight_layout();fig.savefig(output/'figures/04_existing_final_site_metrics.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_2611/task3_step15_full_v5_existing_artifact_audit')
    args=parser.parse_args()
    if args.output.resolve()==args.previous.resolve() or args.previous.resolve() in args.output.resolve().parents:
        raise ValueError('输出目录不能放在历史输入目录内')
    artifacts={};hashes={}
    for region in REGIONS:
        artifacts[region]={}
        for name in ('config.json','training.json','metrics.json'):
            path=args.previous/region/'v5'/name
            if not path.is_file():raise FileNotFoundError(f'缺少已有文件：{path}。本脚本不会自动训练补齐。')
            artifacts[region][name]=json.loads(path.read_text(encoding='utf-8-sig'))
            hashes[f'{region}/v5/{name}']=digest(path)
    existing=args.output/'experiment_metadata.json'
    if args.output.exists():
        if not existing.is_file():raise ValueError('输出目录已存在且不属于本只读审计，请指定新目录')
        previous_meta=json.loads(existing.read_text(encoding='utf-8'))
        if previous_meta.get('audit_mode')!=AUDIT_MODE or previous_meta.get('input_artifact_sha256')!=hashes:
            raise ValueError('已有输出不属于相同历史文件的只读审计，请指定新目录')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    metadata=dict(completed=False,audit_mode=AUDIT_MODE,training_runs=0,models_loaded=False,
        data_sha256_from_historical_config=EXPECTED_HASH,input_artifact_sha256=hashes,
        runner_sha256=digest(Path(__file__)),target_stage_metrics_available=False,
        individual_records_exported=False,
        limitations=['只读历史配置、日志与最终聚合指标，不训练、不加载模型、不读取个人预测文件。',
            '早期来源AUC不是早期目标AUC，不能推断目标地区在哪轮下降。',
            '不同阶段同时改变训练时间、学习率及多项损失，不能归因于某一约束。',
            '训练损失数值及系数不等于梯度强度或因果重要性。',
            '只包含历史种子42，不把它当作跨初始化的稳定性验证。'])
    dump(existing,metadata)
    schedules=[];curves=[];phases=[];checks=[];final=[]
    for region in REGIONS:
        files=artifacts[region]
        result=audit_fold(region,files['config.json'],files['training.json'],files['metrics.json'])
        for collection,values in zip((schedules,curves,phases,checks,final),result):collection.extend(values)
        print(f'任务3 第15步：已审计{region}历史V5，零训练，记录{len(files["training.json"]["history"])}轮',flush=True)
    tables=dict(loss_schedule=pd.DataFrame(schedules),source_training_curves=pd.DataFrame(curves),
        source_phase_summary=pd.DataFrame(phases),artifact_consistency_checks=pd.DataFrame(checks),
        historical_final_metrics=pd.DataFrame(final))
    for name,table in tables.items():table.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(tables['loss_schedule'],tables['source_training_curves'],tables['historical_final_metrics'],args.output)
    failed=tables['artifact_consistency_checks'].loc[lambda x:~x.passed]
    metadata.update(completed=True,audit_passed=failed.empty,failed_check_count=len(failed))
    dump(existing,metadata)
    if not failed.empty:print(f'注意：历史文件有{len(failed)}项一致性检查未通过，已记录；不会重训。',flush=True)
    print(f'任务3 第15步完成：训练0次，模型加载0次，输出目录：{args.output}',flush=True)


if __name__=='__main__':main()
