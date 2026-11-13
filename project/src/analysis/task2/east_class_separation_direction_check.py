"""任务2 第19步：合并核对同类集中、异类区分与既有模型排序错误。

不训练、不加载模型；复用第17步距离规则、历史划分和已有配对错误汇总。
只分析总体、人口、健康功能及人口健康组合，不做单变量重要性排名。
距离差只是改进方向线索，不是因果效应、最优分类上限或信息充分性证明。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import east_distance_reference_robustness as robustness

previous=robustness.previous
audit=previous.audit
SPACES={
    'full_19':tuple(audit.FEATURES),
    'demographic':robustness.BLOCKS['demographic'],
    'health_function':robustness.BLOCKS['health_function'],
    'demographic_health':robustness.BLOCKS['demographic']+robustness.BLOCKS['health_function'],
}
METRICS=('between_distance','positive_within','negative_within','within_distance',
         'separation','relative_separation')


def separation_summary(raw,y,spec,weights):
    """任务2 第19步：同类距离排除自身配对，两类同类均值各占一半。

    异类距离减去同类平均距离，用来区分整体收缩与标签相关分离。
    相对差再除以异类距离，仅作尺度敏感性对照，不视为分类能力指标。
    """
    pos=raw[y==1];neg=raw[y==0];np_,nn=len(pos),len(neg)
    if min(np_,nn)<2:return None
    cross=previous.distances(pos,neg,spec,weights)[0]
    pp=previous.distances(pos,pos,spec,weights)[0]
    qq=previous.distances(neg,neg,spec,weights)[0]
    pmean=float(pp[~np.eye(np_,dtype=bool)].mean())
    nmean=float(qq[~np.eye(nn,dtype=bool)].mean())
    between=float(cross.mean());within=(pmean+nmean)/2;gap=between-within
    return dict(positive_n=np_,negative_n=nn,between_distance=between,
                positive_within=pmean,negative_within=nmean,within_distance=within,
                separation=gap,relative_separation=gap/between if between>0 else np.nan)


def error_summary(part,cohort,positive_n,negative_n):
    """任务2 第19步：直接读取已有汇总，不重新运行模型或导出配对。

    第1距离组界限固定；其余组合并。并列评分按半个错误计数。
    完整样本只要求原始19项输入全部非缺失，不重新填补或干预。
    """
    q=part.sort_values('bin')
    if cohort=='all':
        counts=q.pair_count.to_numpy(float)
        losses=q.error_contribution.to_numpy(float)*float(counts.sum())
    else:
        counts=q.complete_pair_count.to_numpy(float)
        losses=counts*q.complete_pair_error_rate.fillna(0).to_numpy(float)
    total=float(counts.sum())
    if not np.isclose(total,positive_n*negative_n):raise ValueError('既有配对数与当前划分不一致')
    if total==0:return None
    near=counts[0];far=float(counts[1:].sum())
    near_error=float(losses[0]/near) if near else np.nan
    far_error=float(losses[1:].sum()/far) if far else np.nan
    return dict(auc=1-float(losses.sum()/total),closest_share=float(near/total),
                closest_error=near_error,other_error=far_error,
                close_minus_other_error=near_error-far_error,
                closest_pairs=int(near),other_pairs=int(far))


def collect(frame,previous_dir,rules,old_bins):
    profiles=[];counts=[];errors=[]
    for region in audit.REGIONS:
        folder=previous_dir/region/'v5';spec=rules[region]['distance_spec']
        for stage in ('val','test'):
            part,_,y,sites=previous.load_part(folder,stage,frame)
            raw=part[list(audit.FEATURES)].to_numpy(float)
            for site in sorted(set(sites)):
                site_mask=sites==site;values=raw[site_mask];labels=y[site_mask]
                for cohort in robustness.COHORTS:
                    keep=np.ones(len(labels),dtype=bool) if cohort=='all' else ~np.isnan(values).any(1)
                    v=values[keep];target=labels[keep];np_=int((target==1).sum());nn=int((target==0).sum())
                    context=dict(held_out=region,stage=stage,site=str(site),cohort=cohort)
                    counts.append(dict(**context,n=len(v),positive_n=np_,negative_n=nn,
                                       distance_eligible=min(np_,nn)>=2,error_eligible=min(np_,nn)>=1))
                    for distance in previous.DISTANCES:
                        weights=np.asarray(rules[region]['settings'][distance]['weights'],dtype=float)
                        prior=old_bins[(old_bins.held_out==region)&(old_bins.stage==stage)&
                            (old_bins.site.astype(str)==str(site))&(old_bins.distance==distance)]
                        if min(np_,nn)>=1:
                            if prior.empty:raise ValueError('缺少第17步配对汇总')
                            result=error_summary(prior,cohort,np_,nn)
                            errors.append(dict(**context,distance=distance,positive_n=np_,negative_n=nn,**result))
                        if min(np_,nn)<2:continue
                        for space,features in SPACES.items():
                            selected=np.array([f in features for f in audit.FEATURES]);w=weights*selected;w/=w.sum()
                            result=separation_summary(v,target,spec,w)
                            profiles.append(dict(**context,distance=distance,space=space,**result))
        print(f'任务2 第19步：{region}同类与异类距离、既有错误汇总核对完成，模型加载0次',flush=True)
    return pd.DataFrame(profiles),pd.DataFrame(counts),pd.DataFrame(errors)


def compare(profiles):
    """任务2 第19步：比较各站点均值，不把大量配对当作独立样本。"""
    rows=[]
    for keys,q in profiles.groupby(['held_out','distance','cohort','space'],sort=False):
        source=q[q.stage=='val'];target=q[q.stage=='test']
        for policy in ('equal_site_all','equal_site_min10'):
            ref=source if policy=='equal_site_all' else source[(source.positive_n>=10)&(source.negative_n>=20)]
            if ref.empty or target.empty:continue
            for site,t in list(target.groupby('site'))+[('SITE_MACRO',target)]:
                row=dict(zip(('held_out','distance','cohort','space'),keys))
                row.update(reference=policy,site=str(site),source_sites='|'.join(sorted(ref.site)),
                           source_site_count=len(ref),target_site_count=len(t),
                           source_small_positive_sites=int((ref.positive_n<10).sum()))
                for metric in METRICS:
                    row['source_'+metric]=float(ref[metric].mean())
                    row['target_'+metric]=float(t[metric].mean())
                    row['delta_'+metric]=row['target_'+metric]-row['source_'+metric]
                rows.append(row)
    return pd.DataFrame(rows)


def draw(profiles,comparison,errors,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=comparison[(comparison.distance=='phone_balanced')&(comparison.reference=='equal_site_all')&
                    (comparison.cohort=='all')&(comparison.space=='full_19')&(comparison.site!='SITE_MACRO')]
    fig,ax=plt.subplots(figsize=(9,6))
    for j,region in enumerate(audit.REGIONS):
        q=base[base.held_out==region]
        ax.scatter(q.delta_within_distance,q.delta_between_distance,label=region,color=f'C{j}')
        for r in q.itertuples():ax.annotate(r.site,(r.delta_within_distance,r.delta_between_distance),fontsize=7,xytext=(3,3),textcoords='offset points')
    lim=max(.01,float(base[['delta_within_distance','delta_between_distance']].abs().max().max()))*1.2
    ax.plot([-lim,lim],[-lim,lim],color='grey',ls='--',label='Equal contraction')
    ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6)
    ax.set(xlabel='Within-class distance: target minus source',ylabel='Between-class distance: target minus source',
           title='Below diagonal: smaller between-minus-within separation')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/01_within_vs_between_contraction.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,cohort in zip(axes,robustness.COHORTS):
        q=comparison[(comparison.distance=='phone_balanced')&(comparison.cohort==cohort)&(comparison.site=='SITE_MACRO')]
        for j,(region,policy) in enumerate((r,p) for r in audit.REGIONS for p in ('equal_site_all','equal_site_min10')):
            t=q[(q.held_out==region)&(q.reference==policy)].set_index('space').reindex(SPACES)
            ax.scatter(np.arange(len(SPACES))+(j-2.5)*.045,t.delta_relative_separation,
                       marker='o' if policy=='equal_site_all' else 'x',color=f'C{j//2}',
                       s=28,label=region+' / '+policy,alpha=.8)
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(len(SPACES)),list(SPACES),rotation=30,ha='right');ax.set_title(cohort)
    axes[0].set_ylabel('Relative separation: target minus source');axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
    fig.suptitle('Descriptive class separation; no feature ranking or significance claim')
    fig.tight_layout();fig.savefig(out/'figures/02_group_separation_reference_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,cohort in zip(axes,robustness.COHORTS):
        q=errors[(errors.distance=='phone_balanced')&(errors.stage=='test')&(errors.cohort==cohort)].copy()
        q['name']=q.held_out+':'+q.site;q=q.sort_values('name');x=np.arange(len(q))
        ax.scatter(x-.1,q.closest_error,label='Closest distance bin',s=24)
        ax.scatter(x+.1,q.other_error,label='Other distance bins',s=24)
        ax.set_xticks(x,q.name,rotation=60,ha='right');ax.set_title(cohort);ax.set_ylim(0,1);ax.legend(fontsize=8)
    axes[0].set_ylabel('Pair ordering error; ties count as 0.5')
    fig.suptitle('Saved full-V5 predictions; source-defined distance cut remains fixed')
    fig.tight_layout();fig.savefig(out/'figures/03_close_pairs_and_ranking_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,cohort in zip(axes,robustness.COHORTS):
        p=profiles[(profiles.stage=='test')&(profiles.distance=='phone_balanced')&(profiles.space=='full_19')&(profiles.cohort==cohort)]
        e=errors[(errors.stage=='test')&(errors.distance=='phone_balanced')&(errors.cohort==cohort)]
        q=p.merge(e,on=['held_out','stage','site','cohort','distance'],validate='one_to_one')
        for j,region in enumerate(audit.REGIONS):
            t=q[q.held_out==region];ax.scatter(t.relative_separation,t.auc,color=f'C{j}',label=region)
            for r in t.itertuples():ax.annotate(r.site,(r.relative_separation,r.auc),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set_xlabel('Relative class separation');ax.set_title(cohort);ax.legend(fontsize=8);ax.margins(x=.18,y=.15)
    axes[0].set_ylabel('Saved full-V5 site AUC')
    fig.suptitle('Association check across sites; not a causal fit or performance bound')
    fig.tight_layout();fig.savefig(out/'figures/04_separation_and_existing_auc.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=previous.ROOT/'project/data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=previous.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--previous-step',type=Path,default=previous.ROOT/'outputs/analysis_2611/task2_step17_east_pair_distance_error_decomposition')
    parser.add_argument('--output',type=Path,default=previous.ROOT/'outputs/analysis_2611/task2_step19_east_class_separation_direction_check')
    args=parser.parse_args()
    if audit.digest(args.data)!=audit.EXPECTED_HASH:raise ValueError('输入数据版本不一致')
    metadata=robustness.read_json(args.previous_step/'experiment_metadata.json')
    if not metadata.get('completed') or metadata['input_sha256']!=audit.EXPECTED_HASH:raise ValueError('第17步未完成或数据版本不同')
    rules=robustness.read_json(args.previous_step/'source_distance_rules.json');checks=[]
    for region in audit.REGIONS:
        for name,expected in rules[region]['source_artifacts'].items():
            path=args.previous/region/'v5'/name;actual=audit.digest(path)
            if actual!=expected:raise ValueError('历史文件版本不一致：'+str(path))
            checks.append(dict(held_out=region,artifact=name,sha256=actual,matched=True))
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    info=dict(completed=False,training_runs=0,optimizer_steps=0,model_loads=0,individual_records_exported=False,
              input_sha256=audit.EXPECTED_HASH,runner_sha256=audit.digest(Path(__file__)),
              prerequisite_hashes={n:audit.digest(args.previous_step/n) for n in ('source_distance_rules.json','distance_pair_bins.csv','experiment_metadata.json')},
              spaces=SPACES,source_references=['equal_site_all','equal_site_min10'],
              limitations=[
                  '仅作方向检查，没有新增置信区间、显著性检验、因果效应或特征重要性排名。',
                  '同类距离排除自身；每类至少2人，来源小阳性站点仍不可靠，单列最小人数敏感性分析。',
                  '异类减同类均值及其相对值只描述所选距离下的分离，不等价于最优可分类性；负值不截断。',
                  '距离逐特征相加，人口健康组合不额外估计交互；分离弱不证明输入信息不足，也不证明隐层崩溃。',
                  '缺失任一侧距离记1；完整样本为原始19项均无缺失的选择后人群，不是缺失干预。',
                  '子空间沿用来源类型和尺度，权重在所选组内重新归一；不能与第18步原始分摊数直接相加。',
                  '同类与异类都按站点内比较，站点均值等权；来源筛选改变参照人群，不视为正确答案。',
                  '近距离与其他距离的错误率来自第17步保存汇总；关联不证明相似造成错误。',
                  '重复使用既有数据做探索，改进方向需后续受控模型对照验证。'])
    audit.dump(args.output/'experiment_metadata.json',info)
    frame=previous.read_raw(args.data)
    bins=pd.read_csv(args.previous_step/'distance_pair_bins.csv',dtype={'site':str})
    profiles,counts,errors=collect(frame,args.previous,rules,bins);comparison=compare(profiles)
    tables=dict(class_distance_profiles=profiles,site_case_counts=counts,region_reference_comparison=comparison,
                distance_error_link=errors,artifact_checks=pd.DataFrame(checks))
    for name,table in tables.items():table.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(profiles,comparison,errors,args.output)
    info.update(completed=True,models_used='第17步已有聚合预测结果；未加载模型')
    audit.dump(args.output/'experiment_metadata.json',info)
    print(f'任务2 第19步完成：训练0次、模型加载0次，已输出5张表及4张图。目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
