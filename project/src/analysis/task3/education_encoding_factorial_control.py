"""任务3 第9步：用四组编码对照拆分教育与其他特征的贡献。

复用第8步普通学习的预处理、环境等权损失、优化及来源验证选轮次。
只改变教育与其他特征是否按类别展开；不加不变性惩罚或特征筛选。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sparse_invariance_feature_pilot as base

FEATURES = base.FEATURES
ARMS = ('numeric', 'education_only', 'others_only', 'all_categorical')
LABELS = ('Numeric', 'Education categorical', 'Others categorical', 'All categorical')
SEEDS = (42, 43, 44)
CONTRASTS = {
    'education_given_numeric_others': ('education_only', 'numeric'),
    'education_given_categorical_others': ('all_categorical', 'others_only'),
    'others_given_numeric_education': ('others_only', 'numeric'),
    'others_given_categorical_education': ('all_categorical', 'education_only'),
    'total_encoding_change': ('all_categorical', 'numeric'),
}


def compose(bases, arrays, arm):
    """任务3 第9步：拼接两种来源训练编码；原特征顺序与截距位置固定。"""
    blocks, groups, names, choices = [], [], [], []
    offset = 0
    for i, feature in enumerate(FEATURES):
        categorical = arm == 'all_categorical' or (arm == 'education_only' and feature == 'education') or (arm == 'others_only' and feature != 'education')
        mode = 'categorical_grouped' if categorical else 'numeric'
        index = bases[mode]['groups'][i]
        blocks.append(arrays[mode][:, index])
        groups.append(np.arange(offset, offset + len(index)))
        names.extend(bases[mode]['names'][j] for j in index)
        choices.append(mode)
        offset += len(index)
    return np.column_stack(blocks + [np.ones((len(blocks[0]), 1))]), groups, names, choices


def fast_auc(y, p):
    """任务3 第9步：秩和AUC，正确处理相同分数；仅用于配对抽样加速。"""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(p)[y == 1].sum() - n1*(n1+1)/2)/(n1*n0))


def paired_bootstrap(frame, predictions, repeats):
    """任务3 第9步：同一组人、同一种子配对比较；阳性阴性分别有放回抽样。"""
    rows = []
    for group, subgroup, mask in base.cohort_masks(frame):
        if subgroup not in ('all', 'phone1'):
            continue
        y = frame.label.to_numpy(dtype=int)[mask]
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(pos) or not len(neg):
            continue
        values = {a: np.stack([predictions[(a, s)][mask] for s in SEEDS]) for a in ARMS}
        original = {a: np.mean([fast_auc(y, p) for p in values[a]]) for a in ARMS}
        draws = {a: np.empty(repeats) for a in ARMS}
        rng = np.random.default_rng(26113900)
        for repeat in range(repeats):
            index = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for arm in ARMS:
                draws[arm][repeat] = np.mean([fast_auc(y[index], p[index]) for p in values[arm]])
        for name, (after, before) in CONTRASTS.items():
            low, high = np.quantile(draws[after] - draws[before], [.025, .975])
            rows.append(dict(group=group, subgroup=subgroup, contrast=name,
                             before=before, after=after, auc_change=original[after]-original[before],
                             ci_low=low, ci_high=high, repeats=repeats, n=len(y), positive_n=len(pos)))
    return rows


def education_profile(train, basis, groups, weights, arm, context):
    """任务3 第9步：其他变量固定时教育项的分数贡献；相对来源训练均值居中。"""
    raw = basis['imputer'].transform(train[list(FEATURES)])[:, FEATURES.index('education')]
    rows = []
    mode = 'categorical_grouped' if arm in ('education_only', 'all_categorical') else 'numeric'
    # 此处basis必须对应教育实际使用的编码，避免把类别系数与数值尺度混用。
    mean, scale, valid = basis['education']
    levels = np.unique(raw)
    values = (levels[:, None] == basis['levels']['education'][None, :]).astype(float) if mode == 'categorical_grouped' else levels[:, None]
    contribution = ((values[:, valid]-mean[valid])/scale[valid]) @ weights[groups[FEATURES.index('education')]]
    for level, effect in zip(levels, contribution):
        keep = raw == level
        phone = train.if_use_mobile_phone.to_numpy() == 1
        rows.append({**context, 'category':float(level), 'logit_contribution':float(effect),
                     'source_train_n':int(keep.sum()), 'source_train_phone1_n':int((keep & phone).sum())})
    return rows


def draw(metrics, macro, bootstrap, permutation, profile, output):
    """任务3 第9步：所有图标明人群与口径；误差区间仅来自固定模型配对抽样。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(10, 5))
    means = macro.groupby(['held_out', 'arm']).auc.mean().unstack().reindex(['east','central','west'])
    means.reindex(columns=ARMS).rename(columns=dict(zip(ARMS,LABELS))).plot.bar(ax=ax, rot=0)
    ax.set_ylabel('Equal-site macro AUC'); ax.set_ylim(.45, 1)
    ax.set_title('Four encoding controls: source-only checkpoint selection')
    fig.tight_layout(); fig.savefig(output/'figures/01_encoding_factorial_region_auc.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    east = metrics[(metrics.held_out=='east') & (metrics.stage=='test') & (metrics.subgroup=='phone1') & (metrics.group!='REGION_ALL')]
    means = east.groupby(['group','arm']).auc.mean().unstack().reindex(columns=ARMS)
    means.rename(columns=dict(zip(ARMS,LABELS))).plot.bar(ax=axes[0], rot=0)
    axes[0].set_ylabel('AUC'); axes[0].set_ylim(0,1); axes[0].set_title('East phone1: all four sites')
    axes[0].legend(fontsize=7)
    for j, contrast in enumerate(list(CONTRASTS)[:2]):
        part = bootstrap[(bootstrap.subgroup=='phone1') & (bootstrap.group!='REGION_ALL') & (bootstrap.contrast==contrast)].set_index('group').reindex(means.index)
        axes[1].errorbar(np.arange(len(part))+(j-.5)*.15, part.auc_change,
                        yerr=np.vstack([np.maximum(0,part.auc_change-part.ci_low), np.maximum(0,part.ci_high-part.auc_change)]),
                        fmt='o', capsize=3, label=contrast.replace('_',' '))
    axes[1].axhline(0,color='black',lw=.8); axes[1].set_xticks(range(len(means))); axes[1].set_xticklabels(means.index)
    axes[1].set_ylabel('Paired AUC change (95% bootstrap interval)'); axes[1].set_title('Effect of education encoding'); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'figures/02_east_phone1_encoding_contrasts.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax, site in zip(axes, ('110114','371082')):
        part = permutation[(permutation.held_out=='east') & (permutation.stage=='test') & (permutation.group==site) & (permutation.subgroup=='phone1')]
        vals = part.groupby('arm').auc_drop.mean().reindex(ARMS)
        ax.bar(range(4),vals,color=colors[:4]); ax.set_xticks(range(4)); ax.set_xticklabels(LABELS,rotation=20,ha='right',fontsize=8)
        ax.axhline(0,color='black',lw=.8); ax.set_title(f'{site}: phone1'); ax.set_ylabel('AUC drop after education permutation')
    fig.tight_layout(); fig.savefig(output/'figures/03_education_permutation_control.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    part = profile[profile.held_out=='east']
    for arm, label in zip(ARMS,LABELS):
        vals = part[part.arm==arm].groupby('category').logit_contribution.mean()
        ax.plot(vals.index,vals.values,'o-',label=label)
    ax.axhline(0,color='black',lw=.8); ax.set_xlabel('Education category code (no clinical interpretation)')
    ax.set_ylabel('Centered education contribution to logit'); ax.set_title('Education function learned in source regions (East held out)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'figures/04_source_education_function.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base.PROJECT/'data/data_mission2.csv')
    parser.add_argument('--output', type=Path, default=base.ROOT/'outputs/analysis_2611/task3_step9_education_encoding_factorial_control')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 500 or args.permutations < 10 or args.bootstraps < 200:
        raise ValueError('至少训练500轮、置换10次、配对抽样200次')
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != base.EXPECTED_HASH:
        raise ValueError('固定数据版本不一致，请核对文件')
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'figures').mkdir()
    metadata = dict(completed=False, input_sha256=digest, seeds=SEEDS, split_seed=42,
                    arms=ARMS, steps=args.steps, permutations=args.permutations, bootstraps=args.bootstraps,
                    positive_weight=base.POSITIVE_WEIGHT, ridge=base.RIDGE, environment_weighting='equal_site',
                    selection='source_validation_equal_site_macro_auc_then_weighted_bce',
                    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependency_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                    target_used_for_training_or_selection=False, individual_records_exported=False,
                    limits='四组改变的是类别表达及其参数维数，固定逐系数L2不能分离表达容量与正则化几何。全部类别同时改变的收益不能都归给教育。类别贡献为模型分数项而非因果效应。配对区间条件于固定模型、固定数据划分，三种子不是独立人群；小阳性数组不宜强推断。此前已多次查看东部结果，本实验属于探索性复核。此处无隐层，不检验隐层表征崩溃。')
    base.dump(args.output/'experiment_metadata.json',metadata)
    frame = base.attach_regions(base.read_raw(args.data))
    rows, selections, permutation, profiles, audits, unknown, intervals = [], [], [], [], [], [], []
    for region in ('east','central','west'):
        train, val, target = base.held_out_split(frame,region,seed=42,column=base.REGION)
        if any(set(a.sample_id) & set(b.sample_id) for a,b in ((train,val),(train,target),(val,target))):
            raise RuntimeError('数据划分存在重叠')
        bases, matrices = {}, {}
        for mode in base.ENCODINGS:
            basis, x = base.fit_basis(train,mode)
            xv, uv = base.transform_basis(val,basis); xt, ut = base.transform_basis(target,basis)
            bases[mode] = basis; matrices[mode] = (x,xv,xt)
            for stage, values in (('source_validation',uv),('test',ut)):
                unknown.extend(dict(held_out=region,encoding=mode,stage=stage,**r) for r in values)
        selected, designs = {}, {}
        # 任务3 第9步：四组、三个种子全部完成来源选模，之后才评估留出地区。
        for arm in ARMS:
            designs[arm] = [compose(bases,{m:v[j] for m,v in matrices.items()},arm) for j in range(3)]
            x, groups, names, choices = designs[arm][0]; xv = designs[arm][1][0]
            audits.append(dict(held_out=region,arm=arm,input_columns=x.shape[1]-1,semantic_groups=len(groups),
                               train_n=len(train),validation_n=len(val),target_n=len(target),disjoint=True))
            folder = args.output/region/arm; folder.mkdir(parents=True)
            base.dump(folder/'feature_encoding.json', dict(column_names=names,feature_modes=dict(zip(FEATURES,choices)),
                       imputation_medians=bases['numeric']['imputer'].statistics_.tolist(),
                       levels={k:v.tolist() for k,v in bases['categorical_grouped']['levels'].items()},
                       centers_scales_valid={m:{f:[v.tolist() for v in bases[m][f]] for f in FEATURES} for m in bases},
                       group_indices=[g.tolist() for g in groups]))
            for seed in SEEDS:
                print(f'任务3 第9步：留出{region}，条件={arm}，种子={seed}',flush=True)
                result = base.train_candidate(x,train.label.to_numpy(dtype=int),train[base.ENVIRONMENT].to_numpy(),
                           xv,val.label.to_numpy(dtype=int),val[base.ENVIRONMENT].to_numpy(),groups,seed,0.,None,args.steps)
                selected[(arm,seed)] = result
                selections.append(dict(held_out=region,arm=arm,seed=seed,epoch=result['epoch'],source_macro_auc=result['source_macro'],source_weighted_bce=result['source_loss']))
                base.dump(folder/f'source_selection_{seed}.json',{k:v for k,v in result.items() if k!='weights'})
                np.savez(folder/f'selected_parameters_{seed}.npz',weights=result['weights'],column_names=np.array(names+['intercept']))
        predictions = {}
        for (arm,seed), result in selected.items():
            context = dict(held_out=region,arm=arm,seed=seed)
            weights = result['weights']; groups = designs[arm][0][1]
            edu_mode = 'categorical_grouped' if arm in ('education_only','all_categorical') else 'numeric'
            profiles.extend(education_profile(train,bases[edu_mode],groups,weights,arm,context))
            for stage, part, design in zip(('source_train','source_validation','test'),(train,val,target),designs[arm]):
                values = design[0]; p = expit(values@weights)
                if stage == 'test': predictions[(arm,seed)] = p
                for group,subgroup,keep in base.cohort_masks(part):
                    rows.append(dict(**context,stage=stage,group=group,subgroup=subgroup,**base.aggregate_metrics(part.label.to_numpy(dtype=int)[keep],p[keep])))
                if stage != 'source_train':
                    permutation.extend(dict(**context,stage=stage,**r) for r in base.education_permutation(part,values,weights,groups[FEATURES.index('education')],args.permutations))
        if region == 'east':
            print('任务3 第9步：计算东部固定模型的配对抽样区间',flush=True)
            intervals.extend(paired_bootstrap(target,predictions,args.bootstraps))
        pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(selections).to_csv(args.output/'source_only_selection.csv',index=False,encoding='utf-8-sig')
    table = pd.DataFrame(rows)
    sites = table[(table.stage=='test') & (table.group!='REGION_ALL') & (table.subgroup=='all')]
    macro = sites.groupby(['held_out','arm','seed'],as_index=False)[['auc','auprc','brier']].mean()
    outputs = dict(region_site_macro=macro, global_site_macro=sites.groupby(['arm','seed'],as_index=False)[['auc','auprc','brier']].mean(),
                   encoding_audits=pd.DataFrame(audits),unknown_categories=pd.DataFrame(unknown),
                   education_permutation_repeats=pd.DataFrame(permutation),education_category_contribution=pd.DataFrame(profiles),
                   east_paired_bootstrap=pd.DataFrame(intervals))
    # 任务3 第9步：汇总同种子、同人群的四组差值；不据目标结果选择条件。
    contrasts = []
    for name,(after,before) in CONTRASTS.items():
        keys = ['held_out','seed','stage','group','subgroup']
        a = table[table.arm==after].set_index(keys); b = table[table.arm==before].set_index(keys)
        delta = (a[['auc','auprc','brier']] - b[['auc','auprc','brier']]).reset_index()
        delta['contrast'] = name; contrasts.extend(delta.to_dict('records'))
    outputs['paired_metric_changes'] = pd.DataFrame(contrasts)
    for name, values in outputs.items(): values.to_csv(args.output/f'{name}.csv',index=False,encoding='utf-8-sig')
    draw(table,macro,outputs['east_paired_bootstrap'],outputs['education_permutation_repeats'],outputs['education_category_contribution'],args.output)
    metadata['completed'] = True; base.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第9步完成，输出目录：{args.output}',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
