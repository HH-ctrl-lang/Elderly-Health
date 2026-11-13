"""任务2 第17步：复用完整V5，分解同站点异标签样本的距离与排序错误。

第9步已有失败人群画像和来源阳性覆盖，第11步已有手机组AUC分解。
本步仅补充19维特征相似度分层与相同模型输入的异标签碰撞检查。
不训练、不调参、不导出逐人标签、预测、特征或样本配对。
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT.parent
sys.path.insert(0,str(PROJECT/'src/analysis/task3'))
import full_v5_loss_gradient_audit as audit
from elderly_health.data import ENVIRONMENT,read_raw

PHONE=('if_use_mobile_phone','using_phone_year','hours_phone_per_day')
DISTANCES=('phone_balanced','equal_19')


def distance_spec(frame):
    """任务2 第17步：距离类型和稳健尺度只由来源训练特征确定。"""
    spec=[]
    for feature in audit.FEATURES:
        values=frame[feature].dropna().to_numpy(dtype=float)
        if not len(values):raise ValueError('来源特征整列缺失：'+feature)
        count=len(np.unique(values));scale=float(np.quantile(values,.95)-np.quantile(values,.05))
        spec.append(dict(feature=feature,categorical=count<=8,source_unique_count=count,scale=scale if scale>0 else 1.))
    return spec


def distance_weights(name):
    weights=np.ones(len(audit.FEATURES),dtype=float)
    if name=='phone_balanced':
        for f in PHONE:weights[list(audit.FEATURES).index(f)]=1/len(PHONE)
    elif name!='equal_19':raise ValueError('未知距离定义')
    return weights/weights.sum()


def distances(a,b,spec,weights,paired=False):
    """任务2 第17步：低基数按类别，其余按来源5%至95%范围计算截断距离。

    两侧任一缺失均记距离1，避免把共同缺失当成已知相似；另报完整配对结果。
    paired=True只计算对应行，用于来源分位点抽样，不生成平方矩阵。
    """
    shape=(len(a),) if paired else (len(a),len(b))
    total=np.zeros(shape);missing=np.zeros(shape,dtype=bool)
    for j,rule in enumerate(spec):
        left=a[:,j] if paired else a[:,j,None]
        right=b[:,j] if paired else b[None,:,j]
        unknown=np.isnan(left)|np.isnan(right);missing|=unknown
        value=(left!=right).astype(float) if rule['categorical'] else np.minimum(np.abs(left-right)/rule['scale'],1.)
        total+=weights[j]*np.where(unknown,1.,value)
    return total,missing


def calibrate_cuts(frame,spec,weights,count,seed):
    """任务2 第17步：各来源训练站点等量抽取异标签配对，仅定义距离区间。"""
    rng=np.random.default_rng(seed);samples=[];sites=[]
    for site,part in frame.groupby(ENVIRONMENT,sort=True):
        a=part.loc[part.label==1,list(audit.FEATURES)].to_numpy(float)
        b=part.loc[part.label==0,list(audit.FEATURES)].to_numpy(float)
        if not len(a) or not len(b):continue
        # 重复抽取配对只是估计距离分位点，不视作新患者或独立检验样本。
        ip=rng.integers(len(a),size=count);inn=rng.integers(len(b),size=count)
        samples.append(distances(a[ip],b[inn],spec,weights,paired=True)[0]);sites.append(str(site))
    if not samples:raise ValueError('来源训练站点没有可用异标签配对')
    cuts=np.unique(np.quantile(np.concatenate(samples),[.25,.5,.75])).tolist()
    return cuts,sites


def pair_bins(raw,y,p,sites,spec,weights,cuts):
    """任务2 第17步：同站点阳性与阴性比较，错误率严格还原1-AUC。"""
    rows=[];checks=[]
    for site in sorted(set(sites)):
        mask=sites==site;values=raw[mask];labels=y[mask];scores=p[mask]
        pos=labels==1;neg=labels==0;np_,nn=int(pos.sum()),int(neg.sum())
        if not np_ or not nn:
            checks.append(dict(site=str(site),eligible=False,positive_n=np_,negative_n=nn,identity_error=None));continue
        distance,missing=distances(values[pos],values[neg],spec,weights)
        error=(scores[pos,None]<scores[None,neg]).astype(float)+.5*(scores[pos,None]==scores[None,neg])
        bins=np.searchsorted(cuts,distance,side='left');total=np_*nn;contribution=0.
        for group in range(len(cuts)+1):
            cell=bins==group;n=int(cell.sum());loss=float(error[cell].sum());complete=cell&~missing
            unique_p=int(cell.any(1).sum());unique_n=int(cell.any(0).sum())
            rows.append(dict(site=str(site),bin=group+1,pair_count=n,positive_n=unique_p,negative_n=unique_n,
                small_cell=unique_p<10 or unique_n<20,weight=n/total,error_rate=loss/n if n else None,
                error_contribution=loss/total,missing_pair_fraction=float(missing[cell].mean()) if n else None,
                complete_pair_count=int(complete.sum()),complete_pair_error_rate=float(error[complete].mean()) if complete.any() else None,
                mean_distance=float(distance[cell].mean()) if n else None))
            contribution+=loss/total
        auc=float(roc_auc_score(labels,scores));difference=contribution-(1-auc)
        if abs(difference)>1e-10:raise RuntimeError('距离分组不能还原AUC错误')
        checks.append(dict(site=str(site),eligible=True,positive_n=np_,negative_n=nn,auc=auc,identity_error=difference))
    return rows,checks


def reference_rows(rows):
    """任务2 第17步：来源验证站点等权，保留构成权重与错误贡献的恒等式。"""
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('来源验证无可用站点')
    result=[]
    for group,part in frame.groupby('bin',sort=True):
        weight=float(part.weight.mean());contribution=float(part.error_contribution.mean())
        result.append(dict(bin=int(group),weight=weight,error_contribution=contribution,
            error_rate=contribution/weight if weight>0 else None,source_site_count=len(part)))
    return result


def gap_decomposition(reference,target):
    """任务2 第17步：以来源条件错误率为参照；分解是描述性恒等式，不是因果归因。"""
    source={r['bin']:r for r in reference};source_error=sum(r['error_contribution'] for r in reference)
    target_error=sum(r['error_contribution'] for r in target);details=[];composition=0.;within=0.;supported=True
    for t in target:
        s=source[t['bin']];available=s['error_rate'] is not None or t['weight']==0
        supported&=available;e=s['error_rate'] if s['error_rate'] is not None else 0.
        c=(t['weight']-s['weight'])*e if available else None
        w=t['error_contribution']-t['weight']*e if available else None
        if available:composition+=c;within+=w
        details.append(dict(bin=t['bin'],source_weight=s['weight'],target_weight=t['weight'],
            source_error_rate=s['error_rate'],target_error_rate=t['error_rate'],
            distance_composition=c,within_distance_ordering=w,reference_supported=available))
    gap=target_error-source_error;difference=composition+within-gap if supported else None
    if supported and abs(difference)>1e-10:raise RuntimeError('距离构成和组内排序项未还原总差')
    return details,dict(source_macro_auc=1-source_error,target_auc=1-target_error,error_gap=gap,
        distance_composition=composition if supported else None,within_distance_ordering=within if supported else None,
        target_error_at_source_rates=source_error+composition if supported else None,
        complete_reference_support=supported,identity_error=difference)


def collision_audit(x,y,p,sites):
    """任务2 第17步：同一19维模型输入却异标签的配对，任何确定性评分都只能给同分。

    只报告有限样本内必然同分的惩罚，不当作可泛化的性能上界或错标证据。
    """
    rows=[]
    for site in sorted(set(sites)):
        mask=sites==site;values=x[mask];labels=y[mask];scores=p[mask]
        if len(np.unique(labels))<2:continue
        _,inverse=np.unique(values,axis=0,return_inverse=True);counts=np.bincount(inverse)
        positive=np.bincount(inverse,weights=labels);negative=counts-positive;mixed=(positive>0)&(negative>0)
        forced=float((positive*negative).sum());total=float(labels.sum()*(len(labels)-labels.sum()))
        penalty=.5*forced/total;auc=float(roc_auc_score(labels,scores))
        rows.append(dict(site=str(site),n=len(labels),mixed_input_groups=int(mixed.sum()),
            samples_in_mixed_groups=int(counts[mixed].sum()),forced_tie_pairs=int(forced),
            forced_error_penalty=penalty,finite_sample_auc_ceiling=1-penalty,observed_auc=auc))
    return rows


def load_part(folder,stage,frame):
    with np.load(folder/'splits.npz',allow_pickle=False) as z:
        ids=np.asarray(z['sample_id_'+stage],dtype=np.int64);x=np.asarray(z['X_'+stage],dtype=np.float32)
        y=np.asarray(z['y_'+stage],dtype=float).ravel();sites=z['site_env_'+stage].astype(str)
    if len(ids)!=len(np.unique(ids)) or not np.isin(ids,frame.index).all():raise ValueError('保存的样本顺序与原数据不匹配')
    part=frame.loc[ids]
    if not np.array_equal(part.label.to_numpy(),y) or not np.array_equal(part[ENVIRONMENT].astype(str),sites):raise ValueError('标签或站点与保存划分不一致')
    if x.shape!=(len(ids),len(audit.FEATURES)) or not np.isfinite(x).all():raise ValueError('历史模型输入无效')
    return part,x,y,sites


def score(model,x):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(x),256):values.append(torch.sigmoid(model(torch.as_tensor(x[start:start+256]))[0]).numpy().ravel())
    return np.concatenate(values)


def draw(bins,refs,gaps,collisions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    east=bins[(bins.held_out=='east')&(bins.distance=='phone_balanced')&(bins.stage=='test')]
    ref=refs[(refs.held_out=='east')&(refs.distance=='phone_balanced')]
    for site,part in east.groupby('site',sort=True):
        axes[0].plot(part.bin,part.weight,'o-',label=site);axes[1].plot(part.bin,part.error_rate,'o-',label=site)
    axes[0].plot(ref.bin,ref.weight,'k--',label='Source validation macro')
    axes[1].plot(ref.bin,ref.error_rate,'k--',label='Source validation reference')
    for ax in axes:ax.set_xlabel('Source-defined distance bin: 1 = closest');ax.set_ylim(0,1);ax.legend(fontsize=7)
    axes[0].set_ylabel('Share of positive-negative pairs');axes[1].set_ylabel('Within-bin ranking error')
    fig.suptitle('Full V5: similar inputs vs difficulty of ranking; phone-balanced distance')
    fig.tight_layout();fig.savefig(out/'figures/01_east_distance_composition_and_errors.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.5))
    part=gaps[gaps.distance=='phone_balanced'].copy();part['label']=part.held_out+':'+part.site
    pos=np.arange(len(part));ax.bar(pos,part.distance_composition,label='Distance composition')
    ax.bar(pos,part.within_distance_ordering,bottom=part.distance_composition,label='Within-distance ordering',alpha=.7)
    ax.scatter(pos,part.error_gap,color='black',s=20,label='Total error gap');ax.axhline(0,color='grey',lw=1)
    ax.set_xticks(pos,part.label,rotation=45,ha='right');ax.set_ylabel('Target error minus source macro error');ax.legend(fontsize=8)
    fig.suptitle('Descriptive reference-based decomposition; not a causal attribution')
    fig.tight_layout();fig.savefig(out/'figures/02_all_site_error_gap_decomposition.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,name in zip(axes,('distance_composition','within_distance_ordering')):
        values=gaps.pivot(index=['held_out','site'],columns='distance',values=name).astype(float)
        for region in audit.REGIONS:
            part=values.loc[region];ax.scatter(part.phone_balanced,part.equal_19,label=region)
        finite=values.to_numpy()[np.isfinite(values.to_numpy())];limit=max(.01,float(np.abs(finite).max())) if len(finite) else .01
        ax.plot([-limit,limit],[-limit,limit],'k--',lw=1);ax.axhline(0,color='grey',lw=.5);ax.axvline(0,color='grey',lw=.5)
        ax.set_xlabel('Phone-balanced distance');ax.set_ylabel('Equal weight for 19 features');ax.set_title(name);ax.legend(fontsize=8)
    fig.suptitle('Distance-definition sensitivity; same model and saved splits')
    fig.tight_layout();fig.savefig(out/'figures/03_distance_definition_sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.5));labels=collisions.held_out+':'+collisions.site;pos=np.arange(len(labels))
    ax.scatter(pos,collisions.observed_auc,label='Observed full V5 AUC')
    ax.scatter(pos,collisions.finite_sample_auc_ceiling,marker='_',s=130,label='Ceiling from exactly identical inputs only')
    ax.set_xticks(pos,labels,rotation=45,ha='right');ax.set_ylim(0,1.04);ax.set_ylabel('Within-site AUC');ax.legend(fontsize=8)
    fig.suptitle('Exact input collisions: a loose finite-sample bound, not attainable generalization')
    fig.tight_layout();fig.savefig(out/'figures/04_exact_input_collision_bound.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'project/data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_2611/task2_step17_east_pair_distance_error_decomposition')
    parser.add_argument('--calibration-pairs-per-site',type=int,default=5000)
    args=parser.parse_args()
    if args.calibration_pairs_per_site<100:raise ValueError('每站点至少抽取100个来源参考配对')
    audit.require_file(args.data)
    if audit.digest(args.data)!=audit.EXPECTED_HASH:raise ValueError('data_mission2数据版本不一致')
    for region in audit.REGIONS:
        for name in ('config.json','training.json','best_model.pt','splits.npz'):audit.require_file(args.previous/region/'v5'/name)
    frame=read_raw(args.data);args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    metadata=dict(completed=False,training_runs=0,optimizer_steps=0,model_loads=0,individual_records_exported=False,
        input_sha256=audit.digest(args.data),runner_sha256=audit.digest(Path(__file__)),
        helper_sha256=audit.digest(Path(audit.__file__)),
        code_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (audit.models,audit.losses)},
        torch_version=str(torch.__version__),numpy_version=np.__version__,distances=DISTANCES,
        calibration_pairs_per_site=args.calibration_pairs_per_site,
        prior_work_checked=['doc/实验2610.md','doc/实验2611.md','task2_step9_east_persistent_ranking_failures',
            'task2_step11_east_phone_pairwise_decomposition','task3_step18_full_v5_hsic_continuation_control'],
        limitations=['当前为反复使用东部标签的探索性诊断，不是新的独立验证或因果证明。',
            '距离按来源特征类型与尺度定义，非临床验证的相似度；粗分组不能保证各变量构成真正匹配。',
            '手机三列主分析合为一个等权块，19列等权仅为敏感性对照；距离规则不是特征重要性。',
            '全部比较仅在同一站点内部，来源验证参照为有效站点等权；来源验证参与过历史选模，可能乐观。',
            '来源训练异标签配对的分位点决定区间；目标标签仅用于最终错误分解和碰撞描述。',
            '配对共享患者，配对数不是独立样本量；小组按不同患者人数标记，不进行配对级显著性检验。',
            '任一侧缺失记距离1，另报完整特征配对数量和错误率；未发现差异不能证明没有测量或遗漏变量问题。',
            '构成项与组内排序项是依赖来源参照和分组方式的代数恒等式，不代表原因占比。',
            '精确相同输入的异标签配对不等于错标；其上界仅约束当前有限样本，不能证明模型可达到该成绩。'])
    audit.dump(args.output/'experiment_metadata.json',metadata)
    all_bins=[];all_refs=[];all_gaps=[];all_details=[];all_checks=[];collisions=[];rules={}
    for index,region in enumerate(audit.REGIONS):
        folder=args.previous/region/'v5'
        with torch.random.fork_rng(devices=[]):model,cfg,arrays,info=audit.load_fold(folder,region)
        before=audit.state_hash(model)
        train,_,_,_=load_part(folder,'train',frame);spec=distance_spec(train)
        parts={}
        for stage in ('val','test'):
            raw,x,y,sites=load_part(folder,stage,frame);parts[stage]=(raw,x,y,sites,score(model,x))
        raw,x,y,sites,p=parts['test']
        collisions.extend(dict(held_out=region,**r) for r in collision_audit(x,y,p,sites))
        rules[region]=dict(distance_spec=spec,source_artifacts=info['artifact_sha256'],settings={})
        for name in DISTANCES:
            weights=distance_weights(name);cuts,cal_sites=calibrate_cuts(train,spec,weights,args.calibration_pairs_per_site,26111700+index)
            rules[region]['settings'][name]=dict(cuts=cuts,weights=weights.tolist(),calibration_sites=cal_sites)
            stage_rows={}
            for stage,(raw,x,y,sites,p) in parts.items():
                rows,checks=pair_bins(raw[list(audit.FEATURES)].to_numpy(float),y,p,sites,spec,weights,cuts)
                stage_rows[stage]=rows
                all_bins.extend(dict(held_out=region,distance=name,stage=stage,**r) for r in rows)
                all_checks.extend(dict(held_out=region,distance=name,stage=stage,**r) for r in checks)
            reference=reference_rows(stage_rows['val']);all_refs.extend(dict(held_out=region,distance=name,**r) for r in reference)
            targets=pd.DataFrame(stage_rows['test'])
            for site,part in targets.groupby('site',sort=True):
                details,gap=gap_decomposition(reference,part.to_dict('records'))
                all_details.extend(dict(held_out=region,distance=name,site=site,**r) for r in details)
                all_gaps.append(dict(held_out=region,distance=name,site=site,**gap))
        if before!=audit.state_hash(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('只读计算改变了模型')
        metadata['model_loads']+=1
        print(f'任务2 第17步：{region}距离与排序错误分解完成，未训练或更新模型',flush=True)
    tables=dict(distance_pair_bins=pd.DataFrame(all_bins),source_reference=pd.DataFrame(all_refs),
        site_gap_decomposition=pd.DataFrame(all_gaps),gap_bin_details=pd.DataFrame(all_details),
        auc_identity_checks=pd.DataFrame(all_checks),exact_input_collisions=pd.DataFrame(collisions))
    for name,table in tables.items():table.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    audit.dump(args.output/'source_distance_rules.json',rules)
    draw(tables['distance_pair_bins'],tables['source_reference'],tables['site_gap_decomposition'],tables['exact_input_collisions'],args.output)
    metadata.update(completed=True,model_states_unchanged=True,optimizer_gradients_untouched=True)
    audit.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务2 第17步完成：训练0次，优化0步。输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
