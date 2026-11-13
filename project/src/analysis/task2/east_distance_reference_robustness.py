"""任务2 第18步：一次整合患者重采样、来源参照、缺失值与距离定义复核。

复用第17步的距离规则和历史完整V5，不重新训练或选择模型。
按站点×标签×是否有缺失对患者重采样；共享同一患者的所有配对共同变化。
只导出聚合区间、参照成员与特征块距离，不输出患者或配对记录。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import east_pair_distance_error_decomposition as previous

audit=previous.audit
COHORTS=('all','complete')
POLICIES=('equal_site_all','pair_weighted_all','equal_site_min10','shared_complete_sites')
METRICS=('source_auc','target_auc','auc_gap','distance_composition','within_distance_ordering','closest_share_gap')
BLOCKS={
    'demographic':('residence','age','gender','marital_status','living_alone'),
    'socioeconomic':('education','wealth'),
    'health_function':('chronic_disease','IADL','depression','physical_activity'),
    'sleep':('sleep_quality','sleep_duration'),
    'phone':previous.PHONE,
    'other_behavior':('social_activity','smoke','alcohol'),
}


def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def patient_counts(y,missing,repeats,seed):
    """任务2 第18步：患者为重采样单位，固定标签和缺失分层人数，不抽独立配对。"""
    rng=np.random.default_rng(seed);counts=np.ones((repeats+1,len(y)),dtype=float)
    for label in (0,1):
        for status in (False,True):
            ids=np.flatnonzero((y==label)&(missing==status))
            if len(ids):counts[1:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=repeats)
    return counts


def block_distances(a,b,spec,weights):
    """任务2 第18步：特征块只分摊距离本身，不解释预测贡献或因果作用。"""
    result={name:0. for name in BLOCKS}
    for j,rule in enumerate(spec):
        left=a[:,j,None];right=b[None,:,j];missing=np.isnan(left)|np.isnan(right)
        delta=(left!=right).astype(float) if rule['categorical'] else np.minimum(np.abs(left-right)/rule['scale'],1.)
        value=weights[j]*float(np.where(missing,1.,delta).mean())
        block=next(name for name,features in BLOCKS.items() if rule['feature'] in features)
        result[block]+=value
    return result


def site_arrays(raw,y,p,counts,spec,weights,cuts,keep):
    values=raw[keep];labels=y[keep];scores=p[keep];count=counts[:,keep]
    pos=labels==1;neg=labels==0;np_,nn=int(pos.sum()),int(neg.sum())
    if not np_ or not nn:return None
    distance,_=previous.distances(values[pos],values[neg],spec,weights)
    bins=np.searchsorted(cuts,distance,side='left')
    errors=(scores[pos,None]<scores[None,neg]).astype(float)+.5*(scores[pos,None]==scores[None,neg])
    wp=count[:,pos];wn=count[:,neg];den=wp.sum(1)*wn.sum(1)
    w=[];c=[]
    for group in range(len(cuts)+1):
        mask=(bins==group).astype(float)
        w.append(((wp@mask)*wn).sum(1)/den)
        c.append(((wp@(mask*errors))*wn).sum(1)/den)
    w=np.stack(w,axis=1);c=np.stack(c,axis=1)
    if not np.allclose(w.sum(1),1.,atol=1e-10):raise RuntimeError('患者重采样后的配对权重未归一')
    blocks=block_distances(values[pos],values[neg],spec,weights)
    if not np.isclose(sum(blocks.values()),distance.mean(),atol=1e-12):raise RuntimeError('特征块距离未还原总体距离')
    return dict(w=w,c=c,positive_n=np_,negative_n=nn,n=np_+nn,blocks=blocks)


def interval(values):
    values=np.asarray(values,dtype=float);rep=values[1:];valid=rep[np.isfinite(rep)]
    return dict(point=float(values[0]) if np.isfinite(values[0]) else None,
        lower=float(np.quantile(valid,.025)) if len(valid) else None,
        upper=float(np.quantile(valid,.975)) if len(valid) else None,
        valid_repeats=len(valid),positive_fraction=float((valid>0).mean()) if len(valid) else None)


def aggregate(sites,names,policy):
    weight=np.array([sites[n]['positive_n']*sites[n]['negative_n'] if policy=='pair_weighted_all' else 1. for n in names],dtype=float)
    weight/=weight.sum()
    return dict(w=sum(v*sites[n]['w'] for n,v in zip(names,weight)),
        c=sum(v*sites[n]['c'] for n,v in zip(names,weight)),weights=weight,
        blocks={b:sum(v*sites[n]['blocks'][b] for n,v in zip(names,weight)) for b in BLOCKS})


def contrast(source,target):
    sw,sc=source['w'],source['c'];tw,tc=target['w'],target['c']
    rates=np.divide(sc,sw,out=np.zeros_like(sc),where=sw>0)
    unsupported=((sw<=0)&(tw>1e-15)).any(1)
    composition=((tw-sw)*rates).sum(1);within=(tc-tw*rates).sum(1)
    composition[unsupported]=np.nan;within[unsupported]=np.nan
    source_auc=1-sc.sum(1);target_auc=1-tc.sum(1);gap=source_auc-target_auc
    supported=~unsupported
    if not np.allclose((composition+within)[supported],gap[supported],atol=1e-10):raise RuntimeError('重采样分解未还原AUC差距')
    return dict(source_auc=source_auc,target_auc=target_auc,auc_gap=gap,
        distance_composition=composition,within_distance_ordering=within,closest_share_gap=tw[:,0]-sw[:,0])


def summary_row(context,values):
    row=dict(context)
    for metric,vector in values.items():
        for name,value in interval(vector).items():row[metric+'_'+name]=value
    return row


def cohort_counts(context,y,keep):
    labels=y[keep]
    return dict(**context,n=len(labels),positive_n=int((labels==1).sum()),negative_n=int((labels==0).sum()),
        eligible=len(np.unique(labels))==2,small_positive_group=int((labels==1).sum())<10)


def reference_names(sites,policy,shared):
    if policy=='equal_site_min10':return [n for n,s in sites.items() if s['positive_n']>=10 and s['negative_n']>=20]
    if policy=='shared_complete_sites':return [n for n in sites if n in shared]
    if policy.startswith('leave_out:'):return [n for n in sites if n!=policy.split(':',1)[1]]
    return list(sites)


def fold_analysis(region,index,model,folder,frame,rules,repeats,old_bins):
    parts={};case_rows=[];missing_rows=[];site_summaries=[];members=[];blocks=[];checks=[]
    for stage in ('val','test'):
        part,x,y,sites=previous.load_part(folder,stage,frame);p=previous.score(model,x)
        raw=part[list(audit.FEATURES)].to_numpy(float);missing=np.isnan(raw).any(1)
        parts[stage]=(raw,y,sites,p,missing)
        for si,site in enumerate(sorted(set(sites))):
            mask=sites==site;labels=y[mask];unknown=missing[mask]
            for cohort in COHORTS:
                keep=np.ones(mask.sum(),dtype=bool) if cohort=='all' else ~unknown
                case_rows.append(cohort_counts(dict(held_out=region,stage=stage,site=str(site),cohort=cohort),labels,keep))
            for feature in ('ANY_MISSING',)+tuple(audit.FEATURES):
                is_missing=unknown if feature=='ANY_MISSING' else np.isnan(raw[mask,list(audit.FEATURES).index(feature)])
                for label in (0,1):
                    select=labels==label;n=int(select.sum())
                    missing_rows.append(dict(held_out=region,stage=stage,site=str(site),feature=feature,label_group='positive' if label else 'negative',
                        n=n,missing_n=int(is_missing[select].sum()),missing_fraction=float(is_missing[select].mean()) if n else None))
    # 同一患者的重采样权重在两种距离和完整/缺失敏感性分析之间共享。
    count_cache={}
    for stage,(_,y,sites,_,missing) in parts.items():
        for si,site in enumerate(sorted(set(sites))):
            mask=sites==site
            count_cache[stage,site]=patient_counts(y[mask],missing[mask],repeats,26111800+index*10000+(0 if stage=='val' else 5000)+si)
    outcomes={};boot_summaries=[];bin_refs=[];membership_map={}
    for distance in previous.DISTANCES:
        setting=rules['settings'][distance];weights=np.array(setting['weights']);cuts=setting['cuts'];spec=rules['distance_spec']
        arrays={cohort:{stage:{} for stage in parts} for cohort in COHORTS}
        for stage,(raw,y,sites,p,missing) in parts.items():
            for site in sorted(set(sites)):
                mask=sites==site
                for cohort in COHORTS:
                    keep=np.ones(mask.sum(),dtype=bool) if cohort=='all' else ~missing[mask]
                    values=site_arrays(raw[mask],y[mask],p[mask],count_cache[stage,site],spec,weights,cuts,keep)
                    if values is None:continue
                    arrays[cohort][stage][site]=values
                    context=dict(held_out=region,distance=distance,cohort=cohort,stage=stage,site=str(site),positive_n=values['positive_n'],negative_n=values['negative_n'])
                    site_summaries.append(summary_row(context,dict(auc=1-values['c'].sum(1),closest_share=values['w'][:,0])))
                    for block,value in values['blocks'].items():blocks.append(dict(**context,block=block,distance_contribution=value))
                    if cohort=='all':
                        prior=old_bins[(old_bins.held_out==region)&(old_bins.distance==distance)&(old_bins.stage==stage)&(old_bins.site.astype(str)==str(site))].sort_values('bin')
                        if len(prior)!=len(cuts)+1:raise ValueError('第17步聚合结果缺失，不能静默换参照')
                        difference=max(float(np.max(np.abs(prior.weight.to_numpy()-values['w'][0]))),
                            float(np.max(np.abs(prior.error_contribution.to_numpy()-values['c'][0]))))
                        if difference>1e-10:raise ValueError('当前只读分数或距离规则与第17步不一致')
                        checks.append(dict(held_out=region,distance=distance,stage=stage,site=str(site),max_previous_table_difference=difference))
        shared=set(arrays['all']['val'])&set(arrays['complete']['val'])
        for cohort in COHORTS:membership_map[distance,cohort,'TARGET']=set(arrays[cohort]['test'])
        for cohort in COHORTS:
            sources=arrays[cohort]['val'];targets=arrays[cohort]['test']
            if not sources:continue
            policies=list(POLICIES)+['leave_out:'+str(n) for n in sources]
            for policy in policies:
                names=reference_names(sources,policy,shared)
                membership_map[distance,cohort,policy]=set(names)
                members.append(dict(held_out=region,distance=distance,cohort=cohort,reference=policy,
                    available=bool(names),source_site_count=len(names),source_sites='|'.join(str(n) for n in names),
                    min_source_positive=min(sources[n]['positive_n'] for n in names) if names else None,
                    sites_below_10_positive=sum(sources[n]['positive_n']<10 for n in names)))
                if not names:continue
                source=aggregate(sources,names,policy)
                if policy in POLICIES:
                    for group in range(len(cuts)+1):
                        values=dict(weight=source['w'][:,group],error_contribution=source['c'][:,group])
                        values['error_rate']=np.divide(source['c'][:,group],source['w'][:,group],out=np.full(repeats+1,np.nan),where=source['w'][:,group]>0)
                        bin_refs.append(summary_row(dict(held_out=region,distance=distance,cohort=cohort,reference=policy,bin=group+1),values))
                    for block,value in source['blocks'].items():blocks.append(dict(held_out=region,distance=distance,cohort=cohort,stage='source_reference',site=policy,block=block,distance_contribution=value))
                target_list=[(str(n),t) for n,t in targets.items()] if policy in POLICIES else []
                if targets:target_list.append(('SITE_MACRO',aggregate(targets,list(targets),'equal_site_all')))
                for site,target in target_list:
                    values=contrast(source,target)
                    context=dict(held_out=region,distance=distance,cohort=cohort,reference=policy,site=site,source_site_count=len(names),
                        target_site_count=len(targets) if site=='SITE_MACRO' else 1)
                    boot_summaries.append(summary_row(context,values))
                    if policy in POLICIES:outcomes[distance,cohort,policy,site]=values
    # 成对的完整样本减全部样本对照；来源成员可能改变，记录并明确其人群选择含义。
    cohort_effects=[]
    for distance,cohort,policy,site in list(outcomes):
        if cohort!='all' or (distance,'complete',policy,site) not in outcomes:continue
        values={m:outcomes[distance,'complete',policy,site][m]-outcomes[distance,'all',policy,site][m] for m in METRICS}
        cohort_effects.append(summary_row(dict(held_out=region,distance=distance,reference=policy,site=site,
            contrast='complete_minus_all',same_source_members=membership_map[distance,'all',policy]==membership_map[distance,'complete',policy],
            source_members_fixed_by_design=policy=='shared_complete_sites',
            same_target_members=membership_map[distance,'all','TARGET']==membership_map[distance,'complete','TARGET']),values))
    return dict(bootstrap_case_summary=boot_summaries,patient_site_summary=site_summaries,
        source_reference_membership=members,source_reference_bin_summary=bin_refs,complete_case_sensitivity=cohort_effects,
        site_case_counts=case_rows,missingness_profiles=missing_rows,block_distance_profiles=blocks,previous_result_checks=checks)


def draw(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=tables['bootstrap_case_summary'];primary=summary[(summary.distance=='phone_balanced')&(summary.cohort=='all')]
    display=POLICIES[:3]
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    labels=sorted(set(primary.loc[(primary.reference=='equal_site_all')&(primary.site!='SITE_MACRO'),'held_out']+':'+primary.loc[(primary.reference=='equal_site_all')&(primary.site!='SITE_MACRO'),'site']))
    for ax,metric in zip(axes,('closest_share_gap','auc_gap')):
        for j,policy in enumerate(display):
            p=primary[(primary.reference==policy)&(primary.site!='SITE_MACRO')].copy();p['label']=p.held_out+':'+p.site
            p=p.set_index('label').reindex(labels);v=p[metric+'_point'].to_numpy(float);lo=p[metric+'_lower'].to_numpy(float);hi=p[metric+'_upper'].to_numpy(float)
            pos=np.arange(len(labels))+(j-1)*.2
            ax.vlines(pos,lo,hi,lw=1,color=f'C{j}');ax.scatter(pos,v,s=15,color=f'C{j}',label=policy)
        ax.axhline(0,color='grey',lw=1);ax.set_xticks(np.arange(len(labels)),labels,rotation=55,ha='right');ax.set_title(metric);ax.legend(fontsize=7)
    fig.suptitle('Patient-stratified bootstrap: reference populations differ by policy')
    fig.tight_layout();fig.savefig(out/'figures/01_patient_bootstrap_and_reference_policies.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,region in zip(axes,audit.REGIONS):
        p=primary[(primary.held_out==region)&(primary.site=='SITE_MACRO')&((primary.reference=='equal_site_all')|primary.reference.str.startswith('leave_out:'))].copy()
        p=p.sort_values('reference');pos=np.arange(len(p));v=p.within_distance_ordering_point
        ax.vlines(pos,p.within_distance_ordering_lower,p.within_distance_ordering_upper);ax.scatter(pos,v,s=20)
        ax.set_xticks(pos,p.reference.str.replace('leave_out:','omit ',regex=False),rotation=60,ha='right');ax.axhline(0,color='grey',lw=1);ax.set_title(region)
    axes[0].set_ylabel('Within-distance ordering component')
    fig.suptitle('Leave-one-source-site-out reference sensitivity; no refitting')
    fig.tight_layout();fig.savefig(out/'figures/02_reference_leave_one_site_out.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,region in zip(axes,audit.REGIONS):
        p=summary[(summary.held_out==region)&(summary.distance=='phone_balanced')&(summary.reference=='shared_complete_sites')&(summary.site=='SITE_MACRO')]
        for j,cohort in enumerate(COHORTS):
            q=p[p.cohort==cohort]
            if q.empty:continue
            for x,metric in zip((j*.2,1+j*.2),('source_auc','target_auc')):
                r=q.iloc[0];ax.vlines(x,r[metric+'_lower'],r[metric+'_upper'],color=f'C{j}');ax.scatter([x],[r[metric+'_point']],color=f'C{j}',label=cohort if metric=='source_auc' else None)
        ax.set_xticks([.1,1.1],['Source reference','Target macro']);ax.set_title(region);ax.set_ylim(0,1);ax.legend(fontsize=8)
    axes[0].set_ylabel('Fixed-model AUC')
    fig.suptitle('Complete-input patients are a selected population; shared source-site membership')
    fig.tight_layout();fig.savefig(out/'figures/03_complete_case_and_source_support.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    blocks=tables['block_distance_profiles'];matrices=[]
    for cohort in COHORTS:
        p=blocks[(blocks.held_out=='east')&(blocks.distance=='phone_balanced')&(blocks.cohort==cohort)]
        ref=p[(p.stage=='source_reference')&(p.site=='shared_complete_sites')].set_index('block').distance_contribution
        target=p[p.stage=='test'].pivot(index='site',columns='block',values='distance_contribution').reindex(columns=list(BLOCKS))
        matrices.append(target-ref.reindex(list(BLOCKS)))
    finite=np.concatenate([x.to_numpy(float).ravel() for x in matrices]);finite=finite[np.isfinite(finite)];limit=max(.001,float(np.abs(finite).max())) if len(finite) else .001
    im=None
    for ax,cohort,matrix in zip(axes,COHORTS,matrices):
        if matrix.empty:
            ax.text(.5,.5,'No eligible complete-input sites',transform=ax.transAxes,ha='center');ax.set_title(cohort);continue
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(BLOCKS)),list(BLOCKS),rotation=45,ha='right');ax.set_yticks(range(len(matrix)),matrix.index);ax.set_title(cohort)
    if im is not None:
        color_ax=fig.add_axes([.87,.3,.018,.5])
        fig.colorbar(im,cax=color_ax,label='Target minus source mean distance contribution')
    fig.suptitle('Feature-block distance profiles; negative = closer, not causal importance')
    fig.subplots_adjust(bottom=.3,top=.86,right=.83,wspace=.12);fig.savefig(out/'figures/04_east_feature_block_distance_profiles.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    p=summary[(summary.cohort=='all')&(summary.reference=='equal_site_all')&(summary.site=='SITE_MACRO')]
    for ax,metric in zip(axes,('closest_share_gap','within_distance_ordering')):
        for j,region in enumerate(audit.REGIONS):
            q=p[p.held_out==region].set_index('distance')
            if not set(previous.DISTANCES)<=set(q.index):continue
            x=q.loc['phone_balanced'];y=q.loc['equal_19']
            ax.hlines(y[metric+'_point'],x[metric+'_lower'],x[metric+'_upper'],color=f'C{j}');ax.vlines(x[metric+'_point'],y[metric+'_lower'],y[metric+'_upper'],color=f'C{j}')
            ax.scatter([x[metric+'_point']],[y[metric+'_point']],color=f'C{j}',label=region)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set_title(metric);ax.set_xlabel('Phone-balanced distance');ax.set_ylabel('Equal-19 distance');ax.legend(fontsize=8)
    fig.suptitle('Distance sensitivity; uncertainty conditional on saved models and distance cuts')
    fig.tight_layout();fig.savefig(out/'figures/05_distance_definition_and_intervals.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=previous.ROOT/'project/data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=previous.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--previous-step',type=Path,default=previous.ROOT/'outputs/analysis_2611/task2_step17_east_pair_distance_error_decomposition')
    parser.add_argument('--output',type=Path,default=previous.ROOT/'outputs/analysis_2611/task2_step18_east_distance_reference_robustness')
    parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args()
    if args.bootstrap<100:raise ValueError('患者重采样至少100次')
    for path in (args.data,args.previous_step/'source_distance_rules.json',args.previous_step/'distance_pair_bins.csv',args.previous_step/'experiment_metadata.json'):audit.require_file(path)
    if audit.digest(args.data)!=audit.EXPECTED_HASH:raise ValueError('数据版本不一致')
    old_meta=read_json(args.previous_step/'experiment_metadata.json')
    if not old_meta.get('completed') or old_meta['input_sha256']!=audit.EXPECTED_HASH:raise ValueError('第17步未完成或数据版本不同')
    rules=read_json(args.previous_step/'source_distance_rules.json')
    for region in audit.REGIONS:
        for name,expected in rules[region]['source_artifacts'].items():
            path=args.previous/region/'v5'/name;audit.require_file(path)
            if audit.digest(path)!=expected:raise ValueError('历史文件与第17步不一致：'+str(path))
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    metadata=dict(completed=False,training_runs=0,optimizer_steps=0,model_loads=0,bootstrap_repeats=args.bootstrap,
        individual_records_exported=False,input_sha256=audit.EXPECTED_HASH,runner_sha256=audit.digest(Path(__file__)),
        previous_step_artifact_sha256={p.name:audit.digest(p) for p in (args.previous_step/'source_distance_rules.json',args.previous_step/'distance_pair_bins.csv')},
        helper_sha256={Path(m.__file__).name:audit.digest(Path(m.__file__)) for m in (previous,audit,audit.models,audit.losses)},
        reference_policies=POLICIES,cohorts=COHORTS,torch_version=str(torch.__version__),numpy_version=np.__version__,
        limitations=['固定历史模型、来源距离规则及观察到的站点；区间不包含重训、选模、阈值估计或新站点的不确定性。',
            '按站点×标签×是否缺失分层重采样患者，类别人数及缺失分层人数固定；不将配对当成独立患者。',
            '只有2至3名阳性时，bootstrap也无法补出未观察到的异质性；结合最小阳性数和逐站点参照敏感性判断。',
            '参照等站点、配对数加权、排除小阳性站点对应不同人群与权重，不根据东部结果挑选最佳参照。',
            '缺失完整样本是选择后的人群，AUC变化不能证明缺失填补导致性能差；shared_complete_sites固定两种人群的来源成员。',
            'bootstrap正值比例不是P值，未做因果检验；分解不解释原因占比。',
            '特征块分摊的是距离，不是对AUC的贡献；相似度没有医学因果含义。',
            '同一数据反复探索，单种子条件区间不是独立确认，未进行多重比较显著性宣告。'])
    audit.dump(args.output/'experiment_metadata.json',metadata)
    frame=previous.read_raw(args.data);old_bins=pd.read_csv(args.previous_step/'distance_pair_bins.csv',dtype={'site':str});tables={}
    for index,region in enumerate(audit.REGIONS):
        folder=args.previous/region/'v5'
        with torch.random.fork_rng(devices=[]):model,cfg,arrays,info=audit.load_fold(folder,region)
        before=audit.state_hash(model)
        result=fold_analysis(region,index,model,folder,frame,rules[region],args.bootstrap,old_bins)
        for name,rows in result.items():tables.setdefault(name,[]).extend(rows)
        if before!=audit.state_hash(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('只读检查改变模型')
        metadata['model_loads']+=1
        print(f'任务2 第18步：{region}患者重采样、参照、缺失和特征块检查全部完成，训练0次',flush=True)
    frames={name:pd.DataFrame(rows) for name,rows in tables.items()}
    for name,table in frames.items():table.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(frames,args.output)
    metadata.update(completed=True,model_states_unchanged=True,optimizer_gradients_untouched=True)
    audit.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务2 第18步完成：训练0次，优化0步；已输出整合表格与5张图。目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
