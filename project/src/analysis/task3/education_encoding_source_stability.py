"""任务3 第10步：教育编码收益的来源划分与来源站点敏感性复核。

固定比较全部数值编码与仅教育类别化，不按目标地区结果挑选条件。
更换来源划分；东部固定第9步划分后逐一移除来源县区的训练和验证记录。
复用第8、9步预处理与普通学习；重复结果仅用于训练敏感性分析。
"""
import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import education_encoding_factorial_control as previous

base = previous.base
ARMS = ('numeric', 'education_only')
REGIONS = ('east', 'central', 'west')
SPLIT_SEEDS = (42, 91, 2026, 2611, 7319)
INIT_SEEDS = (42, 43, 44)
SCORES = ('auc', 'auprc', 'brier', 'recall_at_05')
KEYS = ['experiment', 'held_out', 'split_seed', 'excluded_site', 'init_seed']
REFERENCE_CATEGORY = 3.


def make_cases(frame):
    """任务3 第10步：预先列出所有条件；删站点沿用原划分，不重新随机分配其他人。"""
    cases = []
    for region in REGIONS:
        for split_seed in SPLIT_SEEDS:
            train, val, target = base.held_out_split(frame, region, seed=split_seed, column=base.REGION)
            context = dict(experiment='repeated_split', held_out=region,
                           split_seed=split_seed, excluded_site='NONE')
            cases.append((context, train, val, target, None))
    train, val, target = base.held_out_split(frame, 'east', seed=42, column=base.REGION)
    for site in sorted(train[base.ENVIRONMENT].unique()):
        reduced_train = train.loc[train[base.ENVIRONMENT] != site].copy()
        reduced_val = val.loc[val[base.ENVIRONMENT] != site].copy()
        excluded = pd.concat([train.loc[train[base.ENVIRONMENT] == site],
                              val.loc[val[base.ENVIRONMENT] == site]])
        context = dict(experiment='source_site_removal', held_out='east',
                       split_seed=42, excluded_site=str(site))
        cases.append((context, reduced_train, reduced_val, target, excluded))
    return cases


def audit_case(context, train, val, target, excluded):
    """任务3 第10步：检查人员与地区隔离，记录删站点引起的样本量和类别覆盖变化。"""
    parts = [train, val, target] + ([] if excluded is None else [excluded])
    disjoint = all(not (set(a.sample_id) & set(b.sample_id))
                   for i, a in enumerate(parts) for b in parts[i+1:])
    if not disjoint or train.empty or val.empty:
        raise RuntimeError('划分重叠或来源数据为空')
    if train.label.nunique() != 2:
        raise RuntimeError('来源训练必须包含两个标签类别')
    if (train[base.REGION] == context['held_out']).any() or (val[base.REGION] == context['held_out']).any():
        raise RuntimeError('留出地区进入了来源训练或验证')
    if excluded is not None and (train[base.ENVIRONMENT].eq(context['excluded_site']).any()
                                 or val[base.ENVIRONMENT].eq(context['excluded_site']).any()):
        raise RuntimeError('被移除站点仍在训练或验证中')
    result = dict(**context, disjoint=True,
                  training_sites=int(train[base.ENVIRONMENT].nunique()),
                  validation_sites=int(val[base.ENVIRONMENT].nunique()),
                  eligible_validation_sites=sum(part.label.nunique() == 2 for _, part in val.groupby(base.ENVIRONMENT)),
                  excluded_n=0 if excluded is None else len(excluded))
    for name, part in zip(('train', 'validation', 'target'), (train, val, target)):
        result[name+'_n'] = len(part)
        result[name+'_positive_n'] = int(part.label.sum())
    return result


def metric_rows(context, stage, frame, p):
    """任务3 第10步：站点、地区合并与站点宏平均分别记录，禁止混用口径。"""
    rows = []
    for group, subgroup, keep in base.cohort_masks(frame):
        values = base.aggregate_metrics(frame.label.to_numpy(dtype=int)[keep], p[keep])
        rows.append(dict(**context, stage=stage, group=group, subgroup=subgroup,
                         auc_components=int(np.isfinite(values['auc'])), **values))
    table = pd.DataFrame(rows)
    for subgroup in ('all', 'phone1', 'phone2'):
        sites = table[(table.group != 'REGION_ALL') & (table.subgroup == subgroup)]
        if sites.empty:
            continue
        n = int(sites.n.sum()); positive_n = int(sites.positive_n.sum())
        values = {s: float(sites[s].mean()) for s in SCORES}
        rows.append(dict(**context, stage=stage, group='SITE_MACRO', subgroup=subgroup,
                         **values, n=n, positive_n=positive_n, positive_rate=positive_n/n,
                         auc_components=int(sites.auc.notna().sum())))
    return rows


def paired_changes(table):
    """任务3 第10步：同划分、同初始种子、同人群逐项配对；Brier正差表示变差。"""
    keys = KEYS + ['stage', 'group', 'subgroup']
    numeric = table.loc[table.arm == 'numeric'].drop(columns='arm')
    categorical = table.loc[table.arm == 'education_only'].drop(columns='arm')
    pair = numeric.merge(categorical, on=keys, suffixes=('_numeric', '_education'),
                         validate='one_to_one')
    if len(pair) != len(numeric) or len(pair) != len(categorical):
        raise RuntimeError('两模型的人群未完全配对')
    if not np.array_equal(pair.n_numeric, pair.n_education):
        raise RuntimeError('配对模型样本数不同')
    for metric in SCORES:
        pair[metric+'_change'] = pair[metric+'_education'] - pair[metric+'_numeric']
    return pair


def case_effects(pair):
    """任务3 第10步：先平均同一次划分的初始化种子，避免当成独立人群计数。"""
    keys = [k for k in KEYS if k != 'init_seed'] + ['stage', 'group', 'subgroup']
    scores = [s+suffix for s in SCORES for suffix in ('_numeric', '_education', '_change')]
    effects = pair.groupby(keys, as_index=False)[scores].mean()
    counts = pair.groupby(keys, as_index=False).agg(init_runs=('init_seed', 'size'),
                    valid_auc_runs=('auc_change', 'count'), n=('n_numeric', 'first'),
                    positive_n=('positive_n_numeric', 'first'),
                    min_initialization_gain=('auc_change', 'min'),
                    max_initialization_gain=('auc_change', 'max'))
    return effects.merge(counts, on=keys, validate='one_to_one')


def stability_summary(effects):
    """任务3 第10步：汇报跨划分均值、范围与正收益次数，不计算伪独立显著性。"""
    part = effects[(effects.experiment == 'repeated_split') & (effects.stage == 'test')]
    keys = ['held_out', 'group', 'subgroup']
    records = []
    for key, rows in part.groupby(keys):
        record = dict(zip(keys, key))
        for metric in SCORES:
            values = rows[metric+'_change'].dropna()
            record.update({metric+'_mean_change': float(values.mean()),
                           metric+'_min_change': float(values.min()),
                           metric+'_max_change': float(values.max()),
                           metric+'_positive_splits': int((values > 0).sum()),
                           metric+'_valid_splits': len(values)})
        records.append(record)
    return pd.DataFrame(records)


def removal_sensitivity(effects):
    """任务3 第10步：删站点后收益与完整来源基线比较；不能解释为站点因果效应。"""
    keys = ['held_out', 'stage', 'group', 'subgroup']
    full = effects[(effects.experiment == 'repeated_split') & (effects.split_seed == 42)
                   & (effects.stage == 'test')]
    full = full[keys + ['auc_change']].rename(columns={'auc_change':'full_source_auc_gain'})
    removal = effects[(effects.experiment == 'source_site_removal') & (effects.stage == 'test')].merge(
        full, on=keys, validate='many_to_one')
    removal['gain_difference_from_full_source'] = removal.auc_change - removal.full_source_auc_gain
    return removal


def reference_check(table, path, steps):
    """任务3 第10步：若第9步结果在本地，检查原划分两端结果是否复现。"""
    if not path.exists() or steps != 1000:
        reason = '未找到第9步结果' if not path.exists() else '训练步数不同，未执行严格复现'
        return pd.DataFrame([dict(checked=False, passed=False, reason=reason)])
    old = pd.read_csv(path, dtype={'group':str})
    keys = ['held_out', 'arm', 'seed', 'stage', 'group', 'subgroup']
    old = old[old.arm.isin(ARMS) & old.seed.isin(INIT_SEEDS) & old.held_out.isin(REGIONS)]
    now = table[(table.experiment == 'repeated_split') & (table.split_seed == 42)
                & (table.group != 'SITE_MACRO')].rename(columns={'init_seed':'seed'})
    merged = now.merge(old[keys+list(SCORES)], on=keys, suffixes=('_now','_previous'), validate='one_to_one')
    if len(merged) != len(now) or len(merged) != len(old):
        raise RuntimeError('第9步复现检查的行未完整对应，请核对历史结果')
    audits = []
    for metric in SCORES:
        a = merged[metric+'_now'].to_numpy(); b = merged[metric+'_previous'].to_numpy()
        same_missing = np.array_equal(np.isnan(a), np.isnan(b))
        valid = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else 0.
        audits.append(dict(metric=metric, checked=True, passed=bool(same_missing and maximum < 1e-7),
                           compared_rows=len(merged), max_absolute_difference=maximum))
    if not all(r['passed'] for r in audits):
        raise RuntimeError('原划分未复现第9步结果，请保留输出并核查代码、依赖和历史训练设置')
    return pd.DataFrame(audits)


def save_table(path, name, rows):
    pd.DataFrame(rows).to_csv(path/(name+'.csv'), index=False, encoding='utf-8-sig')


def draw(effects, profiles, output):
    """任务3 第10步：每个点先平均初始化；范围和连线表示敏感性，不作置信区间。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    repeated = effects[(effects.experiment == 'repeated_split') & (effects.stage == 'test')]
    east = repeated[(repeated.held_out == 'east') & (repeated.subgroup == 'phone1')]
    sites = sorted(east.loc[~east.group.isin(['REGION_ALL','SITE_MACRO']), 'group'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, site in zip(axes.flat, sites):
        part = east[east.group == site].set_index('split_seed').reindex(SPLIT_SEEDS)
        ax.plot(range(len(SPLIT_SEEDS)), part.auc_change, 'o-')
        ax.axhline(0, color='black', lw=.8)
        ax.set_title(f'{site}: phone1 (positive n={int(part.positive_n.iloc[0])})')
        ax.set_xticks(range(len(SPLIT_SEEDS))); ax.set_xticklabels(SPLIT_SEEDS)
        ax.set_xlabel('Source split seed'); ax.set_ylabel('AUC change: education categorical - numeric')
    fig.suptitle('Repeated source splits: mean across initialization seeds')
    fig.tight_layout(); fig.savefig(output/'figures/01_east_phone1_split_stability.png', dpi=180); plt.close(fig)
    removed = effects[(effects.experiment == 'source_site_removal') & (effects.stage == 'test')]
    full = repeated[(repeated.held_out == 'east') & (repeated.split_seed == 42)].copy()
    full['excluded_site'] = 'NONE'
    combined = pd.concat([full, removed], ignore_index=True)
    row_order = ['NONE'] + sorted(removed.excluded_site.unique())
    matrices = [combined[(combined.subgroup == subgroup) & combined.group.isin(sites)].pivot(
        index='excluded_site', columns='group', values='auc_change').reindex(index=row_order, columns=sites)
        for subgroup in ('phone1', 'all')]
    limit = max(.01, max(float(np.nanmax(np.abs(m.values))) for m in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, subgroup, values in zip(axes, ('phone1','all'), matrices):
        im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_yticks(range(len(row_order))); ax.set_yticklabels(['Full source' if s=='NONE' else s for s in row_order])
        ax.set_title(f'East {subgroup}: education encoding AUC gain')
        ax.set_xlabel('Target site'); ax.set_ylabel('Removed source site (train and validation)')
        for i in range(len(row_order)):
            for j in range(len(sites)):
                value = values.iloc[i,j]
                ax.text(j,i,f'{value:+.3f}' if np.isfinite(value) else 'NA',ha='center',va='center',
                        color='white' if abs(value) > limit*.65 else 'black', fontsize=9)
    fig.subplots_adjust(left=.1, right=.86, wspace=.17, bottom=.1, top=.93)
    cax = fig.add_axes([.89,.2,.018,.6]); fig.colorbar(im, cax=cax, label='AUC change (mean across initialization seeds)')
    fig.savefig(output/'figures/02_east_source_site_removal.png',dpi=180); plt.close(fig)
    macro = repeated[(repeated.group == 'SITE_MACRO') & (repeated.subgroup == 'all')]
    fig, axes = plt.subplots(1,3,figsize=(14,5))
    for ax, metric in zip(axes, ('auc','auprc','brier')):
        for j, region in enumerate(REGIONS):
            values = macro[macro.held_out == region].set_index('split_seed').reindex(SPLIT_SEEDS)[metric+'_change']
            x = j + np.linspace(-.12,.12,len(values))
            ax.scatter(x, values, s=30)
            ax.hlines(values.mean(),j-.2,j+.2,color='black',lw=2)
        ax.axhline(0,color='grey',lw=.8); ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS)
        ax.set_title(f'{metric.upper()} change'+(' (lower is better)' if metric=='brier' else ' (higher is better)'))
        ax.set_ylabel('Education categorical - numeric')
    fig.suptitle('Equal-site macro metrics: each point is one source split; black line is mean')
    fig.tight_layout(); fig.savefig(output/'figures/03_region_metric_stability.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
    profile = profiles[(profiles.held_out == 'east') & (profiles.arm == 'education_only')]
    for ax, experiment in zip(axes, ('repeated_split','source_site_removal')):
        part = profile[profile.experiment == experiment]
        key = 'split_seed' if experiment == 'repeated_split' else 'excluded_site'
        means = part.groupby([key,'category']).relative_to_category3.mean().unstack(0)
        for column in means:
            ax.plot(means.index,means[column], 'o-', alpha=.65, label=str(column), markersize=3)
        baseline = profile[(profile.experiment == 'repeated_split') & (profile.split_seed == 42)].groupby('category').relative_to_category3.mean()
        ax.plot(baseline.index,baseline.values,'k--',lw=2,label='Full source / split 42')
        ax.axhline(0,color='grey',lw=.8); ax.set_xticks(sorted(profile.category.unique()))
        ax.set_xlabel('Education category code'); ax.set_ylabel('Education logit contribution relative to category 3')
        ax.set_title(experiment.replace('_',' ')); ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Source-learned education function: sensitivity analysis, not causal effects')
    fig.tight_layout(); fig.savefig(output/'figures/04_education_function_stability.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--output', type=Path, default=base.ROOT/'outputs/analysis_2611/task3_step10_education_encoding_source_stability')
    parser.add_argument('--reference', type=Path, default=base.ROOT/'outputs/analysis_2611/task3_step9_education_encoding_factorial_control/metrics.csv')
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 500:
        raise ValueError('训练至少500轮；建议保持1000轮以复现第9步')
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != base.EXPECTED_HASH:
        raise ValueError('固定数据版本不一致')
    frame = base.attach_regions(base.read_raw(args.data))
    cases = make_cases(frame)
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'figures').mkdir()
    metadata = dict(completed=False, input_sha256=digest, split_seeds=SPLIT_SEEDS,
        initialization_seeds=INIT_SEEDS, arms=ARMS, regions=REGIONS, steps=args.steps,
        case_count=len(cases), expected_fit_count=len(cases)*len(INIT_SEEDS)*len(ARMS),
        completed_fit_count=0, positive_weight=base.POSITIVE_WEIGHT, ridge=base.RIDGE,
        source_environment_weighting='equal_site',
        source_site_removal='east_only; split42; remove_same_site_from_training_and_validation; keep_other_rows_fixed',
        selection='source_validation_equal_site_macro_auc_then_weighted_bce',
        target_used_for_training_or_selection=False, individual_records_exported=False,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (base, previous)},
        limits='同一目标人群被重复评估，重复划分和初始化不构成独立外部验证，不报告伪独立显著性。删来源站点同时减少样本、改变类别覆盖和环境等权权重，只检查依赖性，不识别站点因果效应。固定逐系数L2下，类别表达容量与正则化几何不能分离。教育类别贡献是模型关联而非因果效应；此处无隐层，不检验表征崩溃。目标地区已多次探索。')
    base.dump(args.output/'experiment_metadata.json',metadata)
    rows, selections, profiles, audits, unknown = [], [], [], [], []
    for number, (case, train, val, target, excluded) in enumerate(cases,1):
        audits.append(audit_case(case,train,val,target,excluded))
        stages = [('source_train',train),('source_validation',val),('test',target)]
        if excluded is not None:
            stages.append(('source_site_holdout',excluded))
        bases, matrices = {}, {}
        for mode in base.ENCODINGS:
            basis, x = base.fit_basis(train,mode)
            bases[mode] = basis; matrices[mode] = [x]
            for stage, part in stages[1:]:
                transformed, unsupported = base.transform_basis(part,basis)
                matrices[mode].append(transformed)
                unknown.extend(dict(**case,encoding=mode,stage=stage,**r) for r in unsupported)
        folder = args.output/case['experiment']/case['held_out']/str(case['split_seed'])/case['excluded_site']
        folder.mkdir(parents=True)
        selected, designs = {}, {}
        # 任务3 第10步：两模型全部完成来源选模，之后才计算任何目标地区指标。
        for arm in ARMS:
            designs[arm] = [previous.compose(bases,{m:a[j] for m,a in matrices.items()},arm) for j in range(len(stages))]
            x, groups, names, choices = designs[arm][0]
            base.dump(folder/(arm+'_encoding.json'),dict(column_names=names,feature_modes=dict(zip(base.FEATURES,choices)),
                imputation_medians=bases['numeric']['imputer'].statistics_.tolist(),
                levels={k:v.tolist() for k,v in bases['categorical_grouped']['levels'].items()},
                centers_scales_valid={m:{f:[a.tolist() for a in bases[m][f]] for f in base.FEATURES} for m in bases},
                group_indices=[g.tolist() for g in groups]))
            for seed in INIT_SEEDS:
                print(f'任务3 第10步：条件{number}/{len(cases)}，留出{case["held_out"]}，划分{case["split_seed"]}，移除{case["excluded_site"]}，模型{arm}，初始化{seed}',flush=True)
                result = base.train_candidate(x,train.label.to_numpy(dtype=int),train[base.ENVIRONMENT].to_numpy(),
                    designs[arm][1][0],val.label.to_numpy(dtype=int),val[base.ENVIRONMENT].to_numpy(),groups,seed,0.,None,args.steps)
                context = dict(**case,arm=arm,init_seed=seed)
                selected[(arm,seed)] = result
                selections.append(dict(**context,epoch=result['epoch'],source_macro_auc=result['source_macro'],source_weighted_bce=result['source_loss']))
                base.dump(folder/f'{arm}_selection_{seed}.json',{k:v for k,v in result.items() if k!='weights'})
                np.savez(folder/f'{arm}_parameters_{seed}.npz',weights=result['weights'],column_names=np.array(names+['intercept']))
        for (arm,seed), result in selected.items():
            context = dict(**case,arm=arm,init_seed=seed); groups = designs[arm][0][1]
            edu_basis = bases['numeric' if arm=='numeric' else 'categorical_grouped']
            curve = previous.education_profile(train,edu_basis,groups,result['weights'],arm,context)
            reference = next((r['logit_contribution'] for r in curve if r['category']==REFERENCE_CATEGORY), np.nan)
            for r in curve:
                r['reference_category_supported'] = bool(np.isfinite(reference))
                r['relative_to_category3'] = r['logit_contribution'] - reference
            profiles.extend(curve)
            for (stage,part), design in zip(stages,designs[arm]):
                rows.extend(metric_rows(context,stage,part,expit(design[0]@result['weights'])))
        for name, values in dict(metrics=rows,source_only_selection=selections,
                                  split_and_removal_audits=audits,unknown_categories=unknown,
                                  education_category_contribution=profiles).items():
            save_table(args.output,name,values)
        metadata['completed_fit_count'] = len(selections)
        base.dump(args.output/'experiment_metadata.json',metadata)
    table = pd.DataFrame(rows); pair = paired_changes(table); effects = case_effects(pair)
    checks = reference_check(table,args.reference,args.steps)
    for name, values in dict(paired_metric_changes=pair,case_effects=effects,
                            stability_summary=stability_summary(effects),
                            source_site_sensitivity=removal_sensitivity(effects),
                            step9_reproduction_audits=checks).items():
        save_table(args.output,name,values)
    draw(effects,pd.DataFrame(profiles),args.output)
    metadata['completed'] = True; base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第10步完成：共训练{len(selections)}个模型，输出目录：{args.output}',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
