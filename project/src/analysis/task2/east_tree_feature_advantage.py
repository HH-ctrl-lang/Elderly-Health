"""任务2 第14步：解释手机编码1组内浅层树相对MLP的特征依赖。

复现原模型；同站点、同手机组内共享置换，输出组内AUC下降。
来源训练拟合类别风险表和最多四叶的评分替身；东部不参与选模或定规则。
评分替身不是原树规则，只有来源验证拟合度达到预设标准才辅助解释。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.tree import DecisionTreeRegressor

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, predict, get_device
from east_local_learnability import train_model
from east_phone_pairwise_decomposition import bootstrap_change
from east_phone_mechanism_control import PHONE, SEEDS
import east_phone_group_weight_control as previous_code

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
POLICIES = ("own_source_auc", "own_source_loss")
GROUPS = {**{f: (f,) for f in FEATURES}, "phone_duration_pair": PHONE[1:],
          "education_IADL": ("education", "IADL"), "age_education_IADL": ("age", "education", "IADL")}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def safe_number(value):
    return float(value) if np.isfinite(value) else None


def rng_for(cohort, repeat):
    code = hashlib.sha256(f"26111014|{cohort}|{repeat}".encode()).digest()
    return np.random.default_rng(int.from_bytes(code[:8], "little"))


def within_site_orders(sites, repeats, cohort):
    """任务2 第14步：置换不读取标签，不把不同来源站点的特征混在一起。"""
    result = []
    for repeat in range(repeats):
        rng = rng_for(cohort, repeat)
        order = np.arange(len(sites))
        for site in sorted(set(sites)):
            ix = np.flatnonzero(sites == site)
            order[ix] = rng.permutation(ix)
        result.append(order)
    return result


def permuted_matrix(x, columns, orders):
    """任务2 第14步：同组多列共享行置换，保持组内组合；不修改原输入。"""
    values = np.repeat(x[None, :, :], len(orders), axis=0)
    for i, order in enumerate(orders):
        values[i][:, columns] = x[order][:, columns]
    return values.reshape(-1, x.shape[1])


def category_keys(frame, columns):
    return frame[list(columns)].apply(lambda col: col.map(lambda x: format(float(x), ".12g"))).agg(" | ".join, axis=1)


def category_associations(source, cohorts):
    """任务2 第14步：共同填补后，来源手机1组学习固定平滑风险，不按东部挑方向。"""
    prior = float(source.label.mean())
    auc_rows, category_rows = [], []
    for name, columns in GROUPS.items():
        keys = category_keys(source, columns)
        stats = source.assign(category=keys).groupby("category").label.agg(["sum", "count"])
        mapping = ((stats["sum"]+20*prior)/(stats["count"]+20)).to_dict()
        for cohort, part in cohorts.items():
            cats = category_keys(part, columns)
            scores = cats.map(mapping).fillna(prior).to_numpy(dtype=float)
            auc_rows.append({"group": cohort, "feature_group": name, "n": len(part), "positive_n": int(part.label.sum()),
                             "source_risk_auc": float(roc_auc_score(part.label, scores)),
                             "unknown_fraction": float((~cats.isin(mapping)).mean())})
            observed = part.assign(category=cats).groupby("category").label.agg(["sum", "count"])
            for category, row in observed.iterrows():
                n, pos = int(row["count"]), int(row["sum"])
                category_rows.append({"group": cohort, "feature_group": name, "category": category, "n": n, "positive_n": pos,
                    "positive_rate": pos/n if n >= 20 else np.nan, "rate_eligible": n >= 20,
                    "source_n": int(stats.loc[category,"count"]) if category in stats.index else 0,
                    "source_positive_n": int(stats.loc[category,"sum"]) if category in stats.index else 0,
                    "source_smoothed_risk": float(mapping.get(category, prior))})
    return pd.DataFrame(auc_rows), pd.DataFrame(category_rows)


def leaf_rules(tree):
    """任务2 第14步：导出原编码尺度上的替身路径，不能称作原集成树的真实规则。"""
    result = []

    def walk(node, path):
        left, right = int(tree.tree_.children_left[node]), int(tree.tree_.children_right[node])
        if left == right:
            result.append({"leaf": node, "conditions": path, "rule": " AND ".join(path) if path else "ALL",
                           "source_training_n": int(tree.tree_.n_node_samples[node]),
                           "surrogate_score": float(tree.tree_.value[node].ravel()[0])})
            return
        feature, threshold = FEATURES[int(tree.tree_.feature[node])], float(tree.tree_.threshold[node])
        walk(left, path+[f"{feature} <= {threshold:.12g}"])
        walk(right, path+[f"{feature} > {threshold:.12g}"])

    walk(0, [])
    return result


def make_mlp_scorer(model, state):
    def score(x):
        model.load_state_dict(state)
        return predict(model, np.ascontiguousarray(x, dtype=np.float32), get_device("cpu"))
    return score


def paired_drops(permutation):
    """任务2 第14步：同次置换配对，树下降减MLP下降；正值仅提示树更依赖该信息。"""
    keys = ["group", "feature_group", "repeat"]
    tree = permutation.loc[permutation.model == "tree", keys+["auc_drop"]].rename(columns={"auc_drop": "tree_drop"})
    joined = permutation.loc[permutation.model == "mlp"].merge(tree, on=keys, validate="many_to_one")
    joined["tree_minus_mlp_drop"] = joined.tree_drop-joined.auc_drop
    return joined


def draw(importance, gaps, fidelity, output):
    """任务2 第14步：图中前八组只由来源验证树置换下降排序决定。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranking = importance.loc[(importance.group == "SOURCE_VALIDATION") & (importance.model == "tree")].sort_values(["mean_auc_drop", "feature_group"], ascending=[False,True], kind="stable")
    order = ranking.feature_group.head(8).tolist()
    sites = sorted(set(importance.group)-{"SOURCE_VALIDATION"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        for offset, model, policy, color in ((-.1,"tree","fixed_tree","tab:green"),(.1,"mlp","own_source_loss","tab:blue")):
            part = importance.loc[(importance.group == site) & (importance.model == model) & (importance.policy == policy)].groupby("feature_group").mean(numeric_only=True).reindex(order)
            ax.barh(np.arange(len(order))+offset, part.mean_auc_drop, height=.2, color=color, label=model)
        ax.axvline(0, color="black", linewidth=.8); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        ax.set_title(site); ax.set_xlabel("Within-phone-1 AUC drop after permutation"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "figures/01_tree_mlp_feature_reliance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, policy, color in ((-.1,"own_source_auc","tab:orange"),(.1,"own_source_loss","tab:blue")):
        for i, site in enumerate(sites):
            row = gaps.loc[(gaps.group == site) & (gaps.policy == policy) & (gaps.block == "P1_N1")]
            ax.vlines(i+offset, row.delta_block_auc_lower, row.delta_block_auc_upper, color=color)
            ax.scatter(np.full(len(row), i+offset), row.delta_block_auc, color=color, label=policy if i == 0 else None)
    ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
    ax.set_ylabel("Within-phone-1 AUC: tree minus MLP"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "figures/02_tree_advantage_paired_intervals.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for offset, group, color in ((-.1,"SOURCE_VALIDATION","gray"),(.1,"371082","tab:blue")):
        part = importance.loc[(importance.group == group) & (importance.model == "tree")].set_index("feature_group").reindex(order)
        axes[0].bar(np.arange(len(order))+offset, part.mean_auc_drop, width=.2, color=color,label=group)
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels(order,rotation=30,ha="right",fontsize=7)
    axes[0].set_ylabel("Tree permutation AUC drop"); axes[0].legend(fontsize=7)
    axes[1].bar(fidelity.group, fidelity.r2, color="tab:green"); axes[1].axhline(.7,color="gray",linestyle="--")
    axes[1].set_ylabel("Surrogate R-squared to original tree scores")
    axes[1].tick_params(axis="x",rotation=30); axes[1].set_title("Interpretation requires source-validation R2 >= 0.7")
    fig.tight_layout(); fig.savefig(output / "figures/03_source_transfer_and_rule_fidelity.png", dpi=180); plt.close(fig)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous",type=Path,default=ROOT / "outputs/analysis_2611/task2_step13_east_phone_group_weight_control")
    parser.add_argument("--tree-reference",type=Path,default=ROOT / "outputs/analysis_2611/task2_step11_east_phone_pairwise_decomposition")
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/analysis_2611/task2_step14_east_tree_feature_advantage")
    parser.add_argument("--permutations",type=int,default=30)
    parser.add_argument("--bootstrap",type=int,default=500)
    args = parser.parse_args()
    if args.permutations < 10 or args.bootstrap < 200:
        raise ValueError("置换至少10次、配对重采样至少200次")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python":platform.python_version(),"torch":torch.__version__,"sklearn":sklearn.__version__}
    if not old["completed"] or old["versions"] != versions or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("请使用第13步相同环境及完整结果")
    tree_meta=json.loads((args.tree_reference / "experiment_metadata.json").read_text(encoding="utf-8"))
    if not tree_meta["completed"] or tree_meta["versions"] != versions or tree_meta["input_sha256"] != EXPECTED_HASH:
        raise ValueError("第11步浅层树参照的数据或环境不一致")
    prior_text = Path(previous_code.__file__).read_text(encoding="utf-8")
    if old["code_sha256"] not in {hashlib.sha256(s.encode()).hexdigest() for s in (prior_text,prior_text.replace("\n","\r\n"))}:
        raise ValueError("第13步代码与已完成结果不一致")
    frame = attach_regions(read_raw(args.data)); frames = held_out_split(frame,"east",seed=42,column=REGION)
    source, validation, target = frames
    if set(source.sample_id)&set(validation.sample_id) or (set(source.sample_id)|set(validation.sample_id))&set(target.sample_id):
        raise RuntimeError("来源训练、验证、东部重叠")
    splits, processor = prepare_splits(frames,environment_column=REGION)
    old_metrics = pd.read_csv(args.previous / "metrics.csv",dtype={"group":str})
    tree_ref = pd.read_csv(args.tree_reference / "auc_pair_blocks.csv",dtype={"group":str})
    args.output.mkdir(parents=True,exist_ok=False); (args.output / "figures").mkdir()
    meta = {"completed":False,"input_sha256":EXPECTED_HASH,"versions":versions,"seeds":SEEDS,
            "source_split_seed":42,"phone_group":1,"permutations":args.permutations,"bootstrap_draws":args.bootstrap,
            "groups":GROUPS,"target_used_for_training_or_selection":False,"individual_records_exported":False,
            "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_scope":"371082手机1组内，来源损失选模MLP vs 固定浅层树；其他站点和来源选模规则并列",
            "limits":"特征打乱可能破坏与其他特征的相关结构，AUC下降不是临床因果效应，也不是独立可加贡献。相关特征可互相替代，单列下降小不代表无用。树基准AUC较高，下降更大可能部分反映可损失空间，不能单独证明MLP未利用该信息。置换重复不是独立患者，标准差只描述置换变化。配对区间固定模型评分，不包括训练不确定性。来源验证参与MLP选模及图中排序，有选择乐观性。类别风险表使用共同来源填补后的编码，未知组合退回来源先验。替身只是最多四叶的近似评分器，不是原集成树规则；来源验证R2低于0.7不得据此解释。东部反复查看，属于探索，不能把微小变化当根因。"}
    dump(args.output / "experiment_metadata.json",meta)
    scorers, audits = {}, []
    for seed in SEEDS:
        print(f"任务2 第14步：复现MLP seed={seed}",flush=True)
        lowest, state_loss, epoch_loss = float("inf"),None,None
        def capture(model, record):
            nonlocal lowest,state_loss,epoch_loss
            if record["val_loss"] < lowest:
                lowest,epoch_loss = float(record["val_loss"]),int(record["epoch"])
                state_loss = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        cfg = TrainConfig(model="baseline",environment_granularity="region",seed=seed,selection_start_epoch=1,device="cpu")
        model,history = fit(splits[0],splits[1],cfg,epoch_callback=capture)
        state_auc = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        original = json.loads((args.previous / "full" / str(seed) / "training.json").read_text(encoding="utf-8"))
        if int(history["best_epoch"]) != original["epochs"]["own_source_auc"] or epoch_loss != original["epochs"]["own_source_loss"]:
            raise RuntimeError("MLP来源选模轮次未复现")
        for policy,state in (("own_source_auc",state_auc),("own_source_loss",state_loss)):
            scorer = make_mlp_scorer(model,state); scorers["mlp",policy,seed] = scorer
            pv,pt = scorer(splits[1].x),scorer(splits[2].x)
            cohorts = [("SOURCE_VALIDATION",validation,pv)] + [(str(site),target.loc[target[ENVIRONMENT]==site],pt[target[ENVIRONMENT].to_numpy()==site]) for site in sorted(target[ENVIRONMENT].unique())]
            for group,part,p in cohorts:
                oldrow = old_metrics.loc[(old_metrics.arm=="full")&(old_metrics.seed==seed)&(old_metrics.policy==policy)&(old_metrics.group==group)]
                if len(oldrow)!=1: raise ValueError("第13步参照缺失或重复")
                actual = metrics(part.label.to_numpy(dtype=int),p)
                delta = {k:float(actual[k]-oldrow.iloc[0][k]) for k in ("auc","auprc","brier")}
                audits.append({"model":"mlp","seed":seed,"policy":policy,"group":group,
                               **{"difference_"+k:v for k,v in delta.items()},"passed":all(abs(v)<=2e-6 for v in delta.values())})
        folder=args.output/str(seed);folder.mkdir();dump(folder/"training.json",{"config":cfg.to_dict(),"training":history,"loss_epoch":epoch_loss})
    print("任务2 第14步：复现浅层树，仅训练一次，不视作三个独立模型",flush=True)
    fitted=train_model(source,validation,"shallow_tree",42)
    if not np.array_equal(fitted[1].transform(source),splits[0].x): raise RuntimeError("两模型预处理不一致")
    def tree_score(x):
        return fitted[0].predict_proba(x)[:,list(fitted[0].classes_).index(1)]
    scorers["tree","fixed_tree",42]=tree_score
    dump(args.output/"tree_training.json",fitted[2])
    imputed=[]
    for part in frames:
        copy=part.copy();copy.loc[:,list(FEATURES)]=processor.imputer.transform(part[list(FEATURES)]);imputed.append(copy)
    cohorts={"SOURCE_VALIDATION":imputed[1].loc[imputed[1][PHONE[0]]==1]}
    cohorts.update({str(site):imputed[2].loc[(imputed[2][ENVIRONMENT]==site)&(imputed[2][PHONE[0]]==1)] for site in sorted(target[ENVIRONMENT].unique())})
    x_cohorts={name:processor.transform(part) for name,part in cohorts.items()}
    baselines,base_rows={},[]
    for name,part in cohorts.items():
        for key,score in scorers.items():
            p=score(x_cohorts[name]);baselines[name,key]=p
            base_rows.append({"group":name,"model":key[0],"policy":key[1],"seed":key[2],**metrics(part.label.to_numpy(dtype=int),p)})
        oldrow=tree_ref.loc[(tree_ref.arm=="full")&(tree_ref.model=="shallow_tree")&(tree_ref.seed==42)&(tree_ref.group==name)&(tree_ref.block=="P1_N1")]
        if len(oldrow)!=1: raise ValueError("第11步浅层树参照缺失或重复")
        difference=float(roc_auc_score(part.label,baselines[name,("tree","fixed_tree",42)])-oldrow.iloc[0].block_auc)
        audits.append({"model":"tree","seed":42,"policy":"fixed_tree","group":name,"difference_auc":difference,"passed":abs(difference)<=2e-6})
    pd.DataFrame(audits).to_csv(args.output/"model_reproduction_audits.csv",index=False,encoding="utf-8-sig")
    if not all(r["passed"] for r in audits): raise RuntimeError("模型未复现，暂停特征机制解释")
    pd.DataFrame(base_rows).to_csv(args.output/"phone1_baseline_metrics.csv",index=False,encoding="utf-8-sig")
    gap_rows=[]
    for name,part in cohorts.items():
        for policy in POLICIES:
            mlp=np.stack([baselines[name,("mlp",policy,seed)] for seed in SEEDS])
            tree=np.repeat(baselines[name,("tree","fixed_tree",42)][None,:],len(SEEDS),axis=0)
            rows=bootstrap_change(part,mlp,tree,args.bootstrap,seed=26111014)
            gap_rows.extend({"group":name,"policy":policy,**r} for r in rows if r["block"]=="P1_N1")
    permutation=[]
    for name,part in cohorts.items():
        x=x_cohorts[name];orders=within_site_orders(part[ENVIRONMENT].to_numpy(),args.permutations,name)
        y=part.label.to_numpy(dtype=int)
        for feature,columns in GROUPS.items():
            print(f"任务2 第14步：{name}，置换{feature}",flush=True)
            indices=[FEATURES.index(f) for f in columns]
            changed=permuted_matrix(x,indices,orders)
            unchanged=np.array_equal(changed,np.tile(x,(args.permutations,1)))
            for key,score in scorers.items():
                original=baselines[name,key];auc=float(roc_auc_score(y,original))
                predictions=np.tile(original,(args.permutations,1)) if unchanged else score(changed).reshape(args.permutations,len(part))
                for repeat,p in enumerate(predictions):
                    permuted_auc=float(roc_auc_score(y,p))
                    permutation.append({"group":name,"feature_group":feature,"model":key[0],"policy":key[1],"seed":key[2],
                                        "repeat":repeat,"positive_n":int(y.sum()),"negative_n":int((y==0).sum()),
                                        "baseline_auc":auc,"permuted_auc":permuted_auc,"auc_drop":auc-permuted_auc})
    permutation=pd.DataFrame(permutation)
    importance=permutation.groupby(["group","feature_group","model","policy","seed"],sort=False).agg(
        mean_auc_drop=("auc_drop","mean"),permutation_sd=("auc_drop","std"),repeat_n=("repeat","nunique")).reset_index()
    comparisons=paired_drops(permutation)
    source_phone1=imputed[0].loc[imputed[0][PHONE[0]]==1]
    associations,categories=category_associations(source_phone1,cohorts)
    # 最多四叶的评分替身只在来源训练组拟合，先检查来源验证拟合度。
    surrogate=DecisionTreeRegressor(max_depth=2,min_samples_leaf=80,random_state=26111014)
    source_tree_p=tree_score(processor.transform(source_phone1))
    surrogate.fit(source_phone1[list(FEATURES)].to_numpy(),source_tree_p)
    rules=leaf_rules(surrogate);fidelity=[];leaves=[]
    for name,part in {"SOURCE_TRAIN":source_phone1,**cohorts}.items():
        values=part[list(FEATURES)].to_numpy();approximation=surrogate.predict(values)
        actual=tree_score(processor.transform(part));assign=surrogate.apply(values)
        fidelity.append({"group":name,"n":len(part),"r2":float(r2_score(actual,approximation)),"mean_absolute_error":float(np.abs(actual-approximation).mean())})
        for rule in rules:
            keep=assign==rule["leaf"];n=int(keep.sum());pos=int(part.loc[keep,"label"].sum())
            leaves.append({"group":name,"leaf":rule["leaf"],"rule":rule["rule"],"n":n,"positive_n":pos,"eligible":n>=20,
                           "positive_rate":pos/n if n>=20 else np.nan,"mean_tree_score":float(actual[keep].mean()) if n>=20 else np.nan})
    source_r2=next(r["r2"] for r in fidelity if r["group"]=="SOURCE_VALIDATION")
    rule_allowed=bool(source_r2>=.7)
    dump(args.output/"surrogate_rules.json",{"max_depth":2,"minimum_source_leaf_n":80,"rules":rules,
         "source_validation_r2":safe_number(source_r2),"source_fidelity_passed":rule_allowed,
         "interpretation":"仅当来源验证R2>=0.7时辅助概括原树评分；规则不是临床效应，也不是原集成树真实规则"})
    tables={"permutation_repeats":permutation,"permutation_importance":importance,"paired_tree_mlp_reliance":comparisons,
            "tree_vs_mlp_paired_intervals":pd.DataFrame(gap_rows),"source_category_risk_auc":associations,
            "category_profiles":categories,"surrogate_fidelity":pd.DataFrame(fidelity),"surrogate_leaf_profiles":pd.DataFrame(leaves)}
    for name,table in tables.items():table.to_csv(args.output/(name+".csv"),index=False,encoding="utf-8-sig")
    order=draw(importance,pd.DataFrame(gap_rows),pd.DataFrame(fidelity),args.output)
    meta.update(completed=True,models_reproduced=True,source_only_display_order=order,surrogate_source_fidelity_passed=rule_allowed)
    dump(args.output/"experiment_metadata.json",meta)
    print(f"任务2 第14步完成：{args.output}")


if __name__=="__main__":
    main()
