"""任务3 第19步：冻结完整V5，比较原始输入与既有隐层的加性校正。

区别于第11/12步轻量双块的探针与端到端旁路：这里保留完整历史V5分数，
只拟合小型凸校正头，使用来源数据选正则化强度，全部选模结束后评估目标。
不重训V5，不删除其历史损失；小校正头是诊断性扩展，不宣称因果解耦成立。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits

import full_v5_loss_gradient_audit as audit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'task2'))
import east_pair_distance_error_decomposition as pairs

ARMS=('bias_only','latent_offset','raw_offset')
RIDGES=(.01,.1,1.)


def extract(model,x):
    """任务3 第19步：冻结模型在评估模式下提取分数与表示，不产生优化梯度。"""
    logits=[];latent=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),256):
            z,_,c,s,_=model(torch.as_tensor(x[start:start+256]))
            logits.append(z.numpy().ravel());latent.append(torch.cat([c,s],dim=1).numpy())
    return dict(raw_offset=np.asarray(x,dtype=float),latent_offset=np.concatenate(latent).astype(float),
                bias_only=np.empty((len(x),0))),np.concatenate(logits).astype(float)


def source_validation(path):
    """任务3 第19步：这一阶段不读取目标特征和标签。"""
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z['X_val'],dtype=np.float32);y=np.asarray(z['y_val'],dtype=float).ravel()
        sites=z['site_env_val'].astype(str)
    if len(x)!=len(y) or len(sites)!=len(y) or set(np.unique(y))!={0.,1.} or not np.isfinite(x).all():
        raise ValueError('来源验证数组无效')
    return x,y,sites


def fit_offset(features,y,offset,ridge,pos_weight):
    """任务3 第19步：固定原V5分数，最小化加权BCE加L2；截距不惩罚。

    均值与尺度仅由来源训练输入确定；不同空间维数不同，不能当作严格容量匹配。
    使用双精度解析梯度优化凸目标，不使用神经网络重训或随机初始化搜索。
    """
    mean=features.mean(0);scale=features.std(0);scale=np.where(scale>1e-8,scale,1.)
    x=(features-mean)/scale;design=np.column_stack([x,np.ones(len(x))])
    y=np.asarray(y,float).ravel();weight=np.where(y==1,pos_weight,1.)
    def objective(theta):
        z=offset+design@theta
        value=float(np.mean(weight*(np.logaddexp(0,z)-y*z))+.5*ridge*np.dot(theta[:-1],theta[:-1]))
        grad=design.T@(weight*(expit(z)-y))/len(y);grad[:-1]+=ridge*theta[:-1]
        return value,grad
    result=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=500,ftol=1e-11,gtol=1e-7))
    if not result.success or not np.isfinite(result.fun):raise RuntimeError('校正头优化未收敛：'+str(result.message))
    return dict(mean=mean,scale=scale,theta=result.x,ridge=ridge,
                iterations=int(result.nit),objective=float(result.fun),gradient_max=float(np.abs(result.jac).max()))


def correction(head,features):
    return ((features-head['mean'])/head['scale'])@head['theta'][:-1]+head['theta'][-1]


def metrics(y,logits):
    """任务3 第19步：AUC/AP使用分数排序，Brier使用对应概率，避免饱和造成额外并列。"""
    two=len(np.unique(y))==2;p=expit(logits)
    return dict(auc=float(roc_auc_score(y,logits)) if two else np.nan,
                auprc=float(average_precision_score(y,logits)) if two else np.nan,
                brier=float(brier_score_loss(y,p)),n=len(y),positive_n=int(np.sum(y)),negative_n=int(np.sum(1-y)))


def grouped_metrics(y,logits,sites):
    rows=[dict(scope='region_pooled',group='REGION_ALL',**metrics(y,logits))]
    per_site=[]
    for site in sorted(set(sites)):
        keep=sites==site;r=dict(scope='site',group=str(site),**metrics(y[keep],logits[keep]));per_site.append(r)
    rows.extend(per_site)
    for name,selected in [('equal_site_macro',per_site),('min10_site_macro',[r for r in per_site if r['positive_n']>=10 and r['negative_n']>=20])]:
        rows.append(dict(scope=name,group='SITE_MACRO',site_count=len(selected),
            **{k:float(np.mean([r[k] for r in selected])) if selected and all(np.isfinite(r[k]) for r in selected) else np.nan for k in ('auc','auprc','brier')}))
    return rows


def source_phase(model,arrays,path,cfg,region,out,min_gain):
    x,y,_=arrays;y=y.ravel();vx,vy,vs=source_validation(path)
    train,offset=extract(model,x);val,voffset=extract(model,vx)
    anchor_auc=float(roc_auc_score(vy,voffset));heads={};selected={};candidates=[];policy=[]
    for arm in ARMS:
        grid=(.1,) if arm=='bias_only' else RIDGES
        best_gain=-np.inf;best_ridge=None;best_head=None
        for ridge in grid:
            head=fit_offset(train[arm],y,offset,ridge,cfg.pos_weight)
            heads[arm,ridge]=head;candidate=voffset+correction(head,val[arm])
            report=grouped_metrics(vy,candidate,vs);gain=report[0]['auc']-anchor_auc
            for r in report:candidates.append(dict(held_out=region,arm=arm,ridge=ridge,
                source_pooled_auc_gain=gain,iterations=head['iterations'],gradient_max=head['gradient_max'],**r))
            if gain>best_gain+1e-12:best_gain=gain;best_ridge=ridge;best_head=head
            np.savez(out/f'{region}_{arm}_ridge{ridge:g}.npz',**head)
        # 原始完整V5是显式候选。仅有极小来源排序提升时保留基线，不用目标选阈值。
        accepted=best_gain>=min_gain if arm!='bias_only' else False
        selected[arm]=best_head if accepted else None
        policy.append(dict(held_out=region,arm=arm,selection='source_pooled_auc',
            minimum_source_auc_gain=min_gain,best_candidate_ridge=best_ridge,best_candidate_gain=best_gain,
            selected='correction' if accepted else 'anchor',source_anchor_auc=anchor_auc,
            selected_ridge=best_ridge if accepted else None,feature_dimensions=train[arm].shape[1],
            reason='达到预先固定的来源排序增益门槛' if accepted else '保留完整V5；固定0.1校正仍单独报告'))
        print(f'任务3 第19步：{region}/{arm}小校正头完成，来源最佳AUC变化={best_gain:+.4f}；V5未重训',flush=True)
    for r in grouped_metrics(vy,voffset,vs):candidates.append(dict(held_out=region,arm='anchor',ridge=np.nan,source_pooled_auc_gain=0.,**r))
    return heads,selected,candidates,policy


def pair_error_rows(raw,y,logits,sites,rules):
    """任务3 第19步：复用来源固定距离界限，对比相同配对的排序错误。"""
    result=[];setting=rules['settings']['phone_balanced']
    for site in sorted(set(sites)):
        keep=sites==site;v=raw[keep];labels=y[keep];scores=logits[keep]
        for cohort in ('all','complete'):
            use=np.ones(len(v),bool) if cohort=='all' else ~np.isnan(v).any(1)
            values=v[use];target=labels[use];pred=scores[use];pos=target==1;neg=target==0
            if not pos.any() or not neg.any():continue
            distance,_=pairs.distances(values[pos],values[neg],rules['distance_spec'],np.asarray(setting['weights']))
            near=np.searchsorted(setting['cuts'],distance,side='left')==0
            error=(pred[pos,None]<pred[None,neg]).astype(float)+.5*(pred[pos,None]==pred[None,neg])
            for name,mask in [('closest',near),('other',~near)]:
                result.append(dict(site=str(site),cohort=cohort,distance_bin=name,pair_count=int(mask.sum()),
                    error_rate=float(error[mask].mean()) if mask.any() else np.nan,
                    positive_n=int(pos.sum()),negative_n=int(neg.sum())))
    return result


def evaluate_target(model,folder,frame,rules,heads,selected,region):
    part,x,y,sites=pairs.load_part(folder,'test',frame)
    features,offset=extract(model,x);raw=part[list(audit.FEATURES)].to_numpy(float)
    variants=[('anchor','anchor',offset,np.zeros(len(offset)))]
    for arm in ARMS:
        head=selected[arm];delta=np.zeros(len(offset)) if head is None else correction(head,features[arm])
        variants.append((arm,'source_selected',offset+delta,delta))
        delta=correction(heads[arm,.1],features[arm]);variants.append((arm,'fixed_ridge_0p1',offset+delta,delta))
    rows=[];errors=[];use=[]
    for arm,endpoint,score,delta in variants:
        context=dict(held_out=region,arm=arm,endpoint=endpoint)
        for r in grouped_metrics(y,score,sites):rows.append(dict(**context,**r))
        for r in pair_error_rows(raw,y,score,sites,rules):errors.append(dict(**context,**r))
        use.append(dict(**context,anchor_score_std=float(offset.std()),correction_std=float(delta.std()),
                        correction_mean=float(delta.mean()),score_std=float(score.std())))
    return rows,errors,use


def effects(metrics_table):
    keys=['held_out','scope','group'];anchor=metrics_table[metrics_table.arm=='anchor'][keys+['auc','auprc','brier']]
    q=metrics_table[metrics_table.arm!='anchor'].merge(anchor,on=keys,suffixes=('','_anchor'),validate='many_to_one')
    for k in ('auc','auprc','brier'):q['delta_'+k]=q[k]-q[k+'_anchor']
    return q


def draw(candidates,effect,errors,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,region in zip(axes,audit.REGIONS):
        p=candidates[(candidates.held_out==region)&(candidates.scope=='region_pooled')]
        for arm in ('latent_offset','raw_offset'):
            q=p[p.arm==arm].sort_values('ridge');ax.plot(q.ridge,q.source_pooled_auc_gain,marker='o',label=arm)
        ax.axhline(.005,color='grey',ls='--',label='Default acceptance gain');ax.axhline(0,color='grey',lw=.6)
        ax.set_xscale('log');ax.set_title(region);ax.set_xlabel('Ridge penalty');ax.legend(fontsize=7)
    axes[0].set_ylabel('Source pooled AUC change vs full V5')
    fig.tight_layout();fig.savefig(out/'figures/01_source_correction_selection.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ax,endpoint in zip(axes,('source_selected','fixed_ridge_0p1')):
        q=effect[(effect.endpoint==endpoint)&(effect.scope=='site')].copy();q['name']=q.held_out+':'+q.group
        q.pivot(index='name',columns='arm',values='delta_auc').plot.bar(ax=ax,rot=45)
        ax.axhline(0,color='grey',lw=.7);ax.set_title(endpoint);ax.set_ylabel('AUC change vs full V5');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figures/02_site_auc_vs_full_v5.png',dpi=170);plt.close(fig)
    keys=['held_out','site','cohort','distance_bin'];base=errors[errors.arm=='anchor'][keys+['error_rate']]
    q=errors[(errors.arm.isin(('raw_offset','latent_offset')))&(errors.endpoint=='fixed_ridge_0p1')].merge(base,on=keys,suffixes=('','_anchor'),validate='many_to_one')
    q=q[(q.held_out=='east')&(q.cohort=='all')].copy();q['delta']=q.error_rate-q.error_rate_anchor;q['name']=q.site+':'+q.distance_bin
    fig,ax=plt.subplots(figsize=(12,4));q.pivot(index='name',columns='arm',values='delta').plot.bar(ax=ax,rot=35)
    ax.axhline(0,color='grey');ax.set_ylabel('Pair error change; negative is better');ax.set_title('East: fixed ridge, same pairs and source-defined cuts')
    fig.tight_layout();fig.savefig(out/'figures/03_east_close_pair_error_changes.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,k in zip(axes,('delta_auc','delta_brier')):
        q=effect[(effect.scope=='equal_site_macro')&(effect.endpoint=='fixed_ridge_0p1')]
        q.pivot(index='held_out',columns='arm',values=k).plot.bar(ax=ax,rot=0)
        ax.axhline(0,color='grey');ax.set_ylabel(k);ax.set_title('AUC higher is better' if k=='delta_auc' else 'Brier lower is better')
    fig.tight_layout();fig.savefig(out/'figures/04_region_ranking_probability_tradeoffs.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=audit.ROOT/'project/data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=audit.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--distance-step',type=Path,default=audit.ROOT/'outputs/analysis_2611/task2_step17_east_pair_distance_error_decomposition')
    parser.add_argument('--output',type=Path,default=audit.ROOT/'outputs/analysis_2611/task3_step19_full_v5_frozen_offset_control')
    args=parser.parse_args();min_gain=.005
    if audit.digest(args.data)!=audit.EXPECTED_HASH:raise ValueError('输入数据版本不同')
    rules=json.loads((args.distance_step/'source_distance_rules.json').read_text(encoding='utf-8-sig'))
    meta=json.loads((args.distance_step/'experiment_metadata.json').read_text(encoding='utf-8-sig'))
    if not meta.get('completed') or meta['input_sha256']!=audit.EXPECTED_HASH:raise ValueError('第17步距离规则无效')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir();(args.output/'heads').mkdir()
    metadata=dict(completed=False,v5_training_runs=0,model_loads=0,head_fits=0,minimum_source_auc_gain=min_gain,
        input_sha256=audit.EXPECTED_HASH,runner_sha256=audit.digest(Path(__file__)),ridge_grid=RIDGES,
        full_v5_frozen=True,individual_records_exported=False,target_access_after_all_source_selection=True,
        limitations=[
            '历史完整V5保持冻结；其历史约束没有删除。新增校正头只用预测损失和L2，不具有原约束的因果或不变性保证。',
            '区别于轻量双块探针或从零旁路训练：本步固定完整V5分数作为偏置，只学习加性修正。',
            '原始19维与隐层64维的有效容量不同；收益不能单独归因于信息恢复或表示崩溃。',
            '数值编码、数据划分不变；本步没有新增类别编码、交互模型或论文算法复现。',
            '仅来源训练拟合；来源合并AUC选正则，0.005固定门槛用于避免追逐极小增益，不是统计显著性阈值。',
            '同时报告固定0.1正则以区分选择影响；不按目标结果选择部署模型。',
            '来源站点宏AUC和阳性至少10的宏AUC只作敏感性报告，不再遍历选模政策。',
            '历史检查点和本次头拟合都使用同一来源训练人群，非交叉拟合；既有来源验证也被多次探索。',
            'AUC/AP按原始分数排序，Brier按sigmoid概率；与历史单精度概率排序可能有极小并列差异。',
            '单历史种子、同一批探索数据；配对错误只描述关联，不能作为独立确认或因果识别。'])
    audit.dump(args.output/'experiment_metadata.json',metadata)
    torch.set_num_threads(2);cache={};source=[];policies=[];artifacts=[]
    for region in audit.REGIONS:
        folder=args.previous/region/'v5'
        for name,expected in rules[region]['source_artifacts'].items():
            if audit.digest(folder/name)!=expected:raise ValueError('历史文件与距离规则不一致：'+str(folder/name))
        with torch.random.fork_rng(devices=[]):model,cfg,arrays,info=audit.load_fold(folder,region)
        before=audit.state_hash(model)
        for p in model.parameters():p.requires_grad_(False)
        heads,selected,candidates,policy=source_phase(model,arrays,folder/'splits.npz',cfg,region,args.output/'heads',min_gain)
        if before!=audit.state_hash(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('冻结V5被改变')
        cache[region]=(model,heads,selected,before);source.extend(candidates);policies.extend(policy)
        artifacts.append(dict(held_out=region,**info));metadata['model_loads']+=1;metadata['head_fits']+=sum(1 for _ in heads)
    audit.dump(args.output/'source_phase_completed.json',dict(completed=True,artifacts=artifacts,policies=policies))
    # 三折全部来源选模完成后，才读取原始数据及目标数组进行诊断。
    frame=pairs.read_raw(args.data);target=[];errors=[];uses=[]
    for region in audit.REGIONS:
        model,heads,selected,before=cache[region]
        rows,err,use=evaluate_target(model,args.previous/region/'v5',frame,rules[region],heads,selected,region)
        if before!=audit.state_hash(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('目标评估改变冻结V5')
        target.extend(rows);errors.extend(err);uses.extend(use)
    candidates=pd.DataFrame(source);metrics_table=pd.DataFrame(target);error_table=pd.DataFrame(errors);effect=effects(metrics_table)
    bias=effect[effect.arm=='bias_only']
    if bias.delta_auc.abs().max()>1e-10:raise RuntimeError('常数概率校正不应改变排序AUC')
    tables=dict(source_candidate_metrics=candidates,source_selection=pd.DataFrame(policies),target_metrics=metrics_table,
                target_effects=effect,target_pair_errors=error_table,correction_use=pd.DataFrame(uses))
    for name,table in tables.items():table.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(candidates,effect,error_table,args.output)
    metadata.update(completed=True,full_v5_states_unchanged=True,full_v5_gradients_untouched=True)
    audit.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第19步完成：V5重训0次，仅拟合{metadata["head_fits"]}个小校正头，已输出4张图。目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
