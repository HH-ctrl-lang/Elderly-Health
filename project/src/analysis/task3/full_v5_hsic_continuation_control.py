"""任务3 第18步：从已有完整V5短程续训，控制核尺度和约束强度。

三组使用同一历史检查点、来源划分、批次顺序、Dropout种子和新建Adam。
全部有效损失从续训第一轮起使用完整权重，不重放历史训练。
全部来源选模结束后才读取目标数据；历史检查点允许作为第0轮候选。
"""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

import full_v5_loss_gradient_audit as audit
import full_v5_hsic_kernel_audit as kernel

ARMS=('continue_original','scale_4','strength_matched')
ENDPOINTS=('source_selected','fixed_end')


def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def state_copy(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def load_source_validation(path):
    # 任务3 第18步：训练阶段只访问来源训练/验证数组，不读取目标特征或标签。
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z['X_val'],dtype=np.float32);y=np.asarray(z['y_val'],dtype=np.float32).ravel()
        train_ids=set(z['sample_id_train']);val_ids=set(z['sample_id_val'])
        if train_ids&val_ids:raise ValueError('来源训练和验证划分重叠')
        if set(z['env_val'].astype(str))-set(z['env_train'].astype(str)):raise ValueError('来源验证包含未知环境')
    if len(x)!=len(y) or x.shape[1]!=len(audit.FEATURES) or not np.isfinite(x).all() or set(np.unique(y))!={0.,1.}:
        raise ValueError('来源验证数组无效')
    return x,y


def probabilities(model,x):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),256):result.append(torch.sigmoid(model(torch.as_tensor(x[start:start+256]))[0]).numpy().ravel())
    return np.concatenate(result)


def validate(model,val,cfg):
    x,y=val;prob=probabilities(model,x)
    # 仅AUC用于选模，平分保留更早的检查点；BCE只记录。
    p=np.clip(prob.astype(float),1e-7,1-1e-7)
    loss=float(-(cfg.pos_weight*y*np.log(p)+(1-y)*np.log1p(-p)).mean())
    return float(roc_auc_score(y,prob)),loss


def calibrate(anchor,cfg,arrays,index,repeats):
    """任务3 第18步：仅用历史检查点的来源训练梯度，匹配两个改动组的局部力度。"""
    x,_,env=arrays;params=tuple(p for name,p in anchor.named_parameters() if name.split('.')[0] in ('gate','encoder_c','encoder_s'))
    batches=audit.source_batches(env,cfg.batch_size,repeats,26111600+index);rows=[]
    before=audit.state_hash(anchor)
    with torch.random.fork_rng(devices=[]):
        anchor.train()
        for repeat,ids in enumerate(batches):
            torch.manual_seed(261116000+index*1000+repeat)
            _,_,zc,zs,_=anchor(torch.as_tensor(x[ids]))
            row=dict(batch=repeat)
            for sigma in (1.,4.):
                value=cfg.lambda_independence*audit.losses.hsic_loss(zc,zs,sigma=sigma)
                grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                norm=float(torch.sqrt(sum(g.square().sum() for g in grad if g is not None)))
                row['gradient_sigma_'+str(int(sigma))]=norm
            rows.append(row)
    anchor.eval()
    if audit.state_hash(anchor)!=before or any(p.grad is not None for p in anchor.parameters()):raise RuntimeError('校准改变了历史模型')
    base=float(np.mean([r['gradient_sigma_1'] for r in rows]));scaled=float(np.mean([r['gradient_sigma_4'] for r in rows]))
    if min(base,scaled)<=1e-12 or not np.isfinite([base,scaled]).all():raise ValueError('来源梯度过小或非有限，不能可靠匹配力度；不会自动扩大搜索')
    multiplier=scaled/base
    return dict(multiplier=multiplier,original_lambda=cfg.lambda_independence,
        matched_lambda=cfg.lambda_independence*multiplier,calibration_model_sha256=before,
        parameter_groups=['gate','encoder_c','encoder_s'],batches=repeats,
        matching_scope='historical_checkpoint_source_train_only; local_not_whole_trajectory',rows=rows)


def continue_arm(anchor,arrays,val,cfg,epochs,lr,index,folder):
    """任务3 第18步：固定轮数、固定学习率，全部六项有效目标均参与续训。"""
    model=copy.deepcopy(anchor);before=audit.state_hash(model);x,y,env=arrays
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=cfg.weight_decay)
    initial_auc,initial_bce=validate(model,val,cfg)
    history=[dict(epoch=0,source_val_auc=initial_auc,source_val_bce=initial_bce,selected=True)]
    best_auc=initial_auc;best_epoch=0;best=state_copy(model);batch_hash=hashlib.sha256()
    with torch.random.fork_rng(devices=[]):
        for epoch in range(1,epochs+1):
            model.train();order=np.random.default_rng(26111800+index*1000+epoch).permutation(len(x));totals={}
            for batch,start in enumerate(range(0,len(x),cfg.batch_size)):
                ids=order[start:start+cfg.batch_size];batch_hash.update(ids.astype(np.int64).tobytes())
                torch.manual_seed(261118000+index*100000+epoch*1000+batch)
                terms=audit.batch_terms(model,torch.as_tensor(x[ids]),torch.as_tensor(y[ids]),torch.as_tensor(env[ids]),cfg)
                total=sum(terms.values())
                if not torch.isfinite(total):raise FloatingPointError('续训出现非有限损失')
                optimizer.zero_grad(set_to_none=True);total.backward();optimizer.step()
                for key,value in terms.items():totals[key]=totals.get(key,0.)+float(value.detach())*len(ids)
            auc,bce=validate(model,val,cfg);selected=auc>best_auc
            if selected:best_auc=auc;best_epoch=epoch;best=state_copy(model)
            history.append(dict(epoch=epoch,source_val_auc=auc,source_val_bce=bce,selected=selected,
                **{'weighted_'+k:v/len(x) for k,v in totals.items()}))
            print(f'任务3 第18步：{folder.parent.name}/{folder.name}续训{epoch}/{epochs}轮，来源验证AUC={auc:.4f}',flush=True)
    optimizer.zero_grad(set_to_none=True);end=state_copy(model)
    torch.save(best,folder/'source_selected.pt');torch.save(end,folder/'fixed_end.pt')
    report=dict(initial_state_sha256=before,best_epoch=best_epoch,epochs=epochs,learning_rate=lr,
        first_candidate=0,selection='source_validation_auc; ties_keep_earliest',full_weights_from_first_step=True,
        batch_order_sha256=batch_hash.hexdigest(),history=history,config=cfg.to_dict(),
        selected_state_sha256=None,end_state_sha256=audit.state_hash(model))
    model.load_state_dict(best);report['selected_state_sha256']=audit.state_hash(model)
    return report


def source_probe(model,arrays,cfg,index,repeats,count):
    """任务3 第18步：所有模型用同一sigma=4评估，避免各用各的尺子。"""
    x,_,env=arrays;rows=[];model.eval();before=audit.state_hash(model)
    for repeat,ids in enumerate(audit.source_batches(env,cfg.batch_size,repeats,26111600+index)):
        with torch.no_grad():
            _,_,zc,zs,_=model(torch.as_tensor(x[ids]));k=kernel.rbf(zc,4.);l=kernel.rbf(zs,4.)
        permutations=kernel.permutations(env[ids],count,261117000+index*1000+repeat)
        pair=kernel.pairing_stats(k,l,permutations,dict(batch=repeat),'original')
        a=kernel.center(k.double().numpy());b=kernel.center(l.double().numpy())
        den=float(np.linalg.norm(a)*np.linalg.norm(b));cka=float((a*b).sum()/den) if den>1e-12 else None
        for r in pair:r.update(centered_kernel_alignment=cka,
            zc_mean_coordinate_variance=float(zc.var(0,unbiased=False).mean()),
            zs_mean_coordinate_variance=float(zs.var(0,unbiased=False).mean()))
        rows.extend(pair)
    if before!=audit.state_hash(model):raise RuntimeError('机制诊断意外改变模型')
    return rows


def metric(y,p):
    two=len(np.unique(y))==2
    return dict(auc=float(roc_auc_score(y,p)) if two else None,
        auprc=float(average_precision_score(y,p)) if two else None,
        brier=float(brier_score_loss(y,p)),n=len(y),positive_rate=float(y.mean()))


def target_metrics(model,path):
    # 任务3 第18步：此函数只在全部三折、三组的来源训练/选模完成后调用。
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z['X_test'],dtype=np.float32);y=np.asarray(z['y_test'],dtype=np.float32).ravel()
        sites=z['site_env_test'].astype(str)
    if len(x)!=len(y) or len(sites)!=len(y) or not np.isfinite(x).all() or not set(np.unique(y))<={0.,1.}:raise ValueError('目标评估数组无效')
    p=probabilities(model,x);rows=[dict(scope='region_pooled',group='REGION_ALL',**metric(y,p))]
    site=[]
    for name in sorted(set(sites)):
        mask=sites==name;site.append(dict(scope='site',group=name,**metric(y[mask],p[mask])))
    rows.extend(site)
    macro={key:float(np.mean([r[key] for r in site])) if all(r[key] is not None for r in site) else None for key in ('auc','auprc','brier')}
    rows.append(dict(scope='equal_site_macro',group='SITE_MACRO',**macro,n=len(y),positive_rate=float(y.mean())))
    return rows


def effect_table(metrics):
    keys=['held_out','endpoint','scope','group'];base=metrics[metrics.arm=='continue_original'][keys+['auc','auprc','brier']]
    result=metrics[metrics.arm.isin(ARMS[1:])].merge(base,on=keys,suffixes=('','_control'),validate='many_to_one')
    for name in ('auc','auprc','brier'):result['delta_'+name]=result[name]-result[name+'_control']
    return result


def draw(history,metrics,mechanism,effects,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,region in zip(axes,audit.REGIONS):
        for arm in ARMS:
            s=history[(history.held_out==region)&(history.arm==arm)]
            ax.plot(s.epoch,s.source_val_auc,label=arm)
        ax.set_title(region);ax.set_xlabel('Additional epoch');ax.legend(fontsize=7)
    axes[0].set_ylabel('Source validation AUC')
    fig.suptitle('Full V5 continuation; initial checkpoint is an eligible candidate')
    fig.tight_layout();fig.savefig(out/'figures/01_source_continuation_and_selection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    summary=mechanism[mechanism.shuffle=='within_environment'].groupby(['held_out','arm','endpoint'],as_index=False).centered_kernel_alignment.mean()
    for ax,region in zip(axes,audit.REGIONS):
        p=summary[summary.held_out==region].pivot(index='arm',columns='endpoint',values='centered_kernel_alignment')
        p.reindex(['anchor']+list(ARMS)).plot.bar(ax=ax,rot=25);ax.set_title(region);ax.legend(fontsize=7)
    axes[0].set_ylabel('Source kernel alignment (sigma=4)')
    fig.suptitle('Association change is not automatically beneficial disentanglement')
    fig.tight_layout();fig.savefig(out/'figures/02_source_association_common_kernel.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ax,endpoint in zip(axes,ENDPOINTS):
        p=effects[(effects.endpoint==endpoint)&(effects.scope=='site')].copy()
        p['site_label']=p.held_out+':'+p.group
        p.pivot(index='site_label',columns='arm',values='delta_auc').plot.bar(ax=ax,rot=45)
        ax.axhline(0,color='black',lw=1);ax.set_title(endpoint);ax.set_ylabel('Site AUC change vs continuation');ax.legend(fontsize=8)
        if p.delta_auc.notna().any() and p.delta_auc.abs().max()<1e-12:
            ax.text(.5,.65,'All AUC changes are zero',transform=ax.transAxes,ha='center')
    fig.tight_layout();fig.savefig(out/'figures/03_site_auc_change_vs_continuation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,region in zip(axes,audit.REGIONS):
        p=metrics[(metrics.held_out==region)&(metrics.scope=='equal_site_macro')]
        p.pivot(index='arm',columns='endpoint',values='auc').reindex(['anchor']+list(ARMS)).plot.bar(ax=ax,rot=25)
        ax.set_title(region);ax.set_ylim(0,1);ax.legend(fontsize=7)
    axes[0].set_ylabel('Target equal-site macro AUC')
    fig.suptitle('Prespecified endpoints; target results never choose checkpoints')
    fig.tight_layout();fig.savefig(out/'figures/04_region_macro_and_endpoint_tradeoffs.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=audit.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=audit.ROOT/'outputs/analysis_2611/task3_step18_full_v5_hsic_continuation_control')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--learning-rate',type=float,default=3e-5)
    parser.add_argument('--calibration-batches',type=int,default=12)
    parser.add_argument('--probe-batches',type=int,default=6)
    parser.add_argument('--permutations',type=int,default=99)
    args=parser.parse_args()
    if args.epochs<1 or min(args.calibration_batches,args.probe_batches)<2 or args.permutations<19 or not 0<args.learning_rate<1:raise ValueError('续训或诊断参数无效')
    if args.previous.resolve()==args.output.resolve() or args.previous.resolve() in args.output.resolve().parents:raise ValueError('不能写入历史目录')
    for region in audit.REGIONS:
        for name in ('config.json','training.json','best_model.pt','splits.npz'):audit.require_file(args.previous/region/'v5'/name)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    hashes={f'{r}/{n}':audit.digest(args.previous/r/'v5'/n) for r in audit.REGIONS for n in ('config.json','training.json','best_model.pt','splits.npz')}
    plan=dict(experiment='task3_step18_full_v5_hsic_continuation_control',epochs=args.epochs,learning_rate=args.learning_rate,
        calibration_batches=args.calibration_batches,probe_batches=args.probe_batches,permutations=args.permutations,
        arms=ARMS,scale_sigma=4.,source_artifact_sha256=hashes,runner_sha256=audit.digest(Path(__file__)),
        helper_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (audit,kernel,audit.models,audit.losses)},
        torch_version=str(torch.__version__),numpy_version=np.__version__)
    signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        if not (args.output/'plan.json').is_file() or read_json(args.output/'plan.json').get('signature')!=signature:
            raise ValueError('输出目录已有不同计划，不覆盖；请指定新目录')
    else:args.output.mkdir(parents=True)
    (args.output/'figures').mkdir(exist_ok=True);audit.dump(args.output/'plan.json',dict(signature=signature,**plan))
    metadata=dict(completed=False,source_phase_complete=False,target_used_for_selection=False,
        train_from_scratch_runs=0,planned_continuation_runs=9,new_continuation_runs=0,reused_continuation_runs=0,
        individual_records_exported=False,signature=signature,
        limitations=['从历史最终完整模型局部续训，不等同于改变整个从头训练过程。',
            '三组均重置Adam并共用较小固定学习率；原约束续训用于控制新增训练时间和优化器重置。',
            '力度仅在历史检查点的来源批次上匹配，不保证整条训练轨迹或各参数组力度相等。',
            '所有六项有效损失均保留并从第一步使用完整权重；原历史零图/零熵配置保持不变。',
            '第0轮可选；固定末轮为预定机制终点，不能根据目标结果改选。',
            '仅历史种子42，来源重复批次不是独立队列；目标微小变化不能视为稳定改进。'])
    audit.dump(args.output/'experiment_metadata.json',metadata)
    histories=[];selections=[];calibrations=[];folds={};controls=[]
    # 任务3 第18步：第一阶段完成全部来源校准、三组续训和来源选模，不访问目标数组。
    for index,region in enumerate(audit.REGIONS):
        with torch.random.fork_rng(devices=[]):anchor,cfg,arrays,info=audit.load_fold(args.previous/region/'v5',region)
        if cfg.hsic_sigma!=1:raise ValueError('本对照要求历史核尺度为1')
        val=load_source_validation(args.previous/region/'v5/splits.npz')
        calibration=calibrate(anchor,cfg,arrays,index,args.calibration_batches)
        for r in calibration['rows']:calibrations.append(dict(held_out=region,**r))
        folder=args.output/region;folder.mkdir(exist_ok=True);audit.dump(folder/'source_calibration.json',calibration)
        configs={'continue_original':cfg,'scale_4':replace(cfg,hsic_sigma=4.),
            'strength_matched':replace(cfg,lambda_independence=calibration['matched_lambda'])}
        reports={}
        for arm in ARMS:
            arm_folder=folder/arm;arm_folder.mkdir(exist_ok=True);done=arm_folder/'source_complete.json'
            if done.is_file():
                report=read_json(done)
                if report['signature']!=signature:raise ValueError('缓存计划不一致')
                for endpoint in ENDPOINTS:
                    path=arm_folder/(endpoint+'.pt');audit.require_file(path)
                    if audit.digest(path)!=report['checkpoint_sha256'][endpoint]:raise ValueError('缓存检查点变化')
                metadata['reused_continuation_runs']+=1
                print(f'任务3 第18步：复用{region}/{arm}已完成的续训',flush=True)
            else:
                report=continue_arm(anchor,arrays,val,configs[arm],args.epochs,args.learning_rate,index,arm_folder)
                report.update(signature=signature,checkpoint_sha256={e:audit.digest(arm_folder/(e+'.pt')) for e in ENDPOINTS})
                audit.dump(done,report);metadata['new_continuation_runs']+=1
            reports[arm]=report
            histories.extend(dict(held_out=region,arm=arm,**r) for r in report['history'])
            selections.append(dict(held_out=region,arm=arm,selected_additional_epoch=report['best_epoch'],
                initial_state_sha256=report['initial_state_sha256'],lambda_independence=configs[arm].lambda_independence,
                sigma=configs[arm].hsic_sigma,batch_order_sha256=report['batch_order_sha256']))
        same_start=len({r['initial_state_sha256'] for r in reports.values()})==1
        same_order=len({r['batch_order_sha256'] for r in reports.values()})==1
        if not same_start or not same_order or audit.state_hash(anchor)!=calibration['calibration_model_sha256']:
            raise RuntimeError('三组起点、来源批次或历史锚点状态不一致')
        controls.append(dict(held_out=region,same_initial_state=same_start,same_batch_order=same_order,
            anchor_unchanged=audit.state_hash(anchor)==calibration['calibration_model_sha256'],all_losses_full_weight=True))
        folds[region]=(anchor,cfg,arrays,info)
    metadata['source_phase_complete']=True;audit.dump(args.output/'experiment_metadata.json',metadata)
    pd.DataFrame(selections).to_csv(args.output/'source_selection.csv',index=False,encoding='utf-8-sig')
    # 任务3 第18步：第二阶段对已锁定终点作机制诊断和目标聚合评估，绝不反向改动选模。
    mechanisms=[];metrics=[]
    for index,region in enumerate(audit.REGIONS):
        anchor,cfg,arrays,info=folds[region]
        candidates=[('anchor','initial',anchor)]
        for arm in ARMS:
            for endpoint in ENDPOINTS:
                model=copy.deepcopy(anchor);state=torch.load(args.output/region/arm/(endpoint+'.pt'),map_location='cpu',weights_only=True)
                model.load_state_dict(state,strict=True);candidates.append((arm,endpoint,model))
        for arm,endpoint,model in candidates:
            mechanisms.extend(dict(held_out=region,arm=arm,endpoint=endpoint,**r) for r in source_probe(model,arrays,cfg,index,args.probe_batches,args.permutations))
            metrics.extend(dict(held_out=region,arm=arm,endpoint=endpoint,**r) for r in target_metrics(model,args.previous/region/'v5/splits.npz'))
        print(f'任务3 第18步：{region}已完成锁定终点的机制和目标评估',flush=True)
    history=pd.DataFrame(histories);mechanism=pd.DataFrame(mechanisms);metric_frame=pd.DataFrame(metrics);effects=effect_table(metric_frame)
    for name,frame in dict(source_history=history,source_mechanism_batches=mechanism,target_metrics=metric_frame,
        effects_vs_continuation=effects,source_calibration_batches=pd.DataFrame(calibrations),control_checks=pd.DataFrame(controls)).items():
        frame.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    summary=mechanism.groupby(['held_out','arm','endpoint','shuffle'],as_index=False).agg(
        mean_hsic=('hsic','mean'),mean_relative_excess=('relative_excess','mean'),
        mean_kernel_alignment=('centered_kernel_alignment','mean'),zc_variance=('zc_mean_coordinate_variance','mean'),
        zs_variance=('zs_mean_coordinate_variance','mean'))
    summary.to_csv(args.output/'source_mechanism_summary.csv',index=False,encoding='utf-8-sig')
    draw(history,metric_frame,mechanism,effects,args.output)
    metadata['completed']=True;audit.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第18步完成：从头训练0次，本次新增续训{metadata["new_continuation_runs"]}组，复用{metadata["reused_continuation_runs"]}组。输出：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
