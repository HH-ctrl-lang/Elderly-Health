"""任务2 第16步：来源五折外预测复核教育修正，并重复教育打乱对照。

每位来源训练样本的修正拟合分数由未训练过自己的MLP给出。
折内预处理只拟合四折训练数据；各折仅用原独立来源验证集选模。
最终仍修正原完整来源MLP，不使用东部训练、定规则或挑选置换。
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
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, read_raw, attach_regions, held_out_split, prepare_splits
from elderly_health.training import fit, predict, get_device
from elderly_health.evaluation import metrics
from east_phone_mechanism_control import PHONE, SEEDS
from east_phone_pairwise_decomposition import block_metrics, bootstrap_change
import east_education_residual_control as previous
from east_education_residual_control import design_spec, correction_design, fit_correction, apply_correction, dump

EXPECTED_HASH = previous.EXPECTED_HASH
POLICIES = previous.POLICIES
ARMS = ("full", "apparent_phone", "apparent_shared", "oof_phone", "oof_shared", "oof_by_phone")
CONTRASTS = (("oof_shared", "full"), ("oof_shared", "oof_phone"),
             ("oof_shared", "apparent_shared"), ("oof_by_phone", "oof_shared"))
FOLD_SEED = 26111016
SHUFFLE_SEED = 26111600


def make_folds(source):
    """任务2 第16步：按结局×手机组分层，仅为保证折内类别支持，不导出逐人分折。"""
    strata = source.label.astype(int).astype(str)+"|"+source[PHONE[0]].astype(str)
    if strata.value_counts().min() < 5:
        raise ValueError("来源类别×手机组不足5例，不能执行固定五折")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    folds = list(splitter.split(np.zeros(len(source)), strata))
    seen = np.zeros(len(source), dtype=int)
    for train, held in folds:
        if np.intersect1d(train, held).size or len(train)+len(held) != len(source):
            raise RuntimeError("折内训练与留出存在重叠或遗漏")
        seen[held] += 1
    if not np.all(seen == 1):
        raise RuntimeError("来源样本未恰好获得一次折外预测")
    return folds


def shuffle_source(source, repeat):
    """任务2 第16步：同站点×手机组内打乱教育，保留标签和原MLP输入。"""
    result = source.copy()
    rng = np.random.default_rng(SHUFFLE_SEED+repeat)
    for _, index in source.groupby([ENVIRONMENT, PHONE[0]], sort=True).groups.items():
        result.loc[index, "education"] = rng.permutation(source.loc[index, "education"].to_numpy())
    return result


def fit_scores(splits, seed, folder):
    """任务2 第16步：两种来源选模共享一次轨迹；折外标签不进入fit或回调。"""
    lowest, loss_state, loss_epoch = float("inf"), None, None
    def capture(model, record):
        nonlocal lowest, loss_state, loss_epoch
        if record["val_loss"] < lowest:
            lowest, loss_epoch = float(record["val_loss"]), int(record["epoch"])
            loss_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed,
                      selection_start_epoch=1, device="cpu")
    model, history = fit(splits[0], splits[1], cfg, epoch_callback=capture)
    auc_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    result = {}
    for policy, state in (("own_source_auc", auc_state), ("own_source_loss", loss_state)):
        model.load_state_dict(state)
        result[policy] = [predict(model, split.x, get_device("cpu")).astype(float) for split in splits]
    folder.mkdir(parents=True)
    dump(folder/"training.json", {"config": cfg.to_dict(), "training": history,
         "epochs": {"own_source_auc": int(history["best_epoch"]), "own_source_loss": loss_epoch}})
    return result


def aggregate_scores(frame, p, context):
    """任务2 第16步：只返回汇总指标和四块AUC，不返回个人评分。"""
    row = {**context, **metrics(frame.label.to_numpy(dtype=int), p)}
    blocks, _ = block_metrics(frame, p)
    return row, [{**context, **block} for block in blocks]


def shuffle_summary(real_blocks, shuffled_blocks):
    """任务2 第16步：先跨三种子平均，再描述50种打乱结果；不是患者置信区间。"""
    keys = ["policy", "group", "block"]
    real = real_blocks.loc[real_blocks.arm == "oof_shared"].groupby(keys, as_index=False).block_auc.mean()
    shuffled = shuffled_blocks.groupby(keys+["repeat"], as_index=False).block_auc.mean()
    rows = []
    for key, part in shuffled.groupby(keys, sort=True):
        match = real
        for field,value in zip(keys,key):
            match = match.loc[match[field] == value]
        value = float(match.iloc[0].block_auc)
        values = part.block_auc.to_numpy(dtype=float)
        if not np.isfinite(value) or not np.isfinite(values).all():
            continue
        rows.append({**dict(zip(keys,key)), "real_auc": value, "shuffle_mean_auc": float(values.mean()),
                     "shuffle_lower_auc": float(np.quantile(values,.025)),
                     "shuffle_upper_auc": float(np.quantile(values,.975)),
                     "real_minus_shuffle_mean": float(value-values.mean()), "shuffle_repeats": len(values),
                     "shuffle_at_least_real_fraction": float(np.mean(values >= value))})
    return pd.DataFrame(rows)


def draw(blocks, intervals, shuffled, metric_table, output):
    """任务2 第16步：图一复核收益，图二重复打乱，图三全站与来源代价。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = sorted(set(blocks.group)-{"SOURCE_TRAIN", "SOURCE_VALIDATION", "EAST_ALL"})
    fig, axes = plt.subplots(2,2,figsize=(14,9),sharey=True)
    displayed = ("full", "apparent_shared", "oof_phone", "oof_shared", "oof_by_phone")
    for ax,site in zip(axes.ravel(),sites):
        part = blocks.loc[(blocks.group == site)&(blocks.policy == "own_source_loss")&(blocks.block == "P1_N1")]
        for i,arm in enumerate(displayed):
            values = part.loc[part.arm == arm].sort_values("seed").block_auc
            ax.scatter(i+np.linspace(-.07,.07,len(values)), values, color=f"C{i}")
        ax.axhline(.5,color="gray",linestyle="--");ax.set_title(site)
        ax.set_xticks(range(len(displayed)));ax.set_xticklabels(displayed,rotation=25,ha="right",fontsize=7)
        ax.set_ylabel("Within-phone-1 AUC; three seeds")
    fig.tight_layout();fig.savefig(output/"figures/01_crossfit_vs_apparent_correction.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,policy in zip(axes,POLICIES):
        for i,site in enumerate(sites):
            row=shuffled.loc[(shuffled.group==site)&(shuffled.policy==policy)&(shuffled.block=="P1_N1")].iloc[0]
            ax.vlines(i,row.shuffle_lower_auc,row.shuffle_upper_auc,color="gray",linewidth=5,alpha=.5)
            ax.scatter(i,row.shuffle_mean_auc,color="gray",label="Shuffled mean and 95% range" if i==0 else None)
            ax.scatter(i,row.real_auc,color="tab:blue",label="Real education" if i==0 else None)
        ax.set_xticks(range(len(sites)));ax.set_xticklabels(sites);ax.set_title(policy)
        ax.set_ylabel("OOF-corrected phone-1 AUC");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"figures/02_repeated_education_shuffle.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    groups=["SOURCE_TRAIN","SOURCE_VALIDATION"]+sites
    for j,arm in enumerate(("full","apparent_shared","oof_shared")):
        part=blocks.loc[(blocks.arm==arm)&(blocks.policy=="own_source_loss")&(blocks.block=="P1_N1")]
        values=part.groupby("group").block_auc.mean().reindex(groups)
        axes[0].plot(range(len(groups)),values,marker="o",label=arm,color=f"C{j}")
    axes[0].set_xticks(range(len(groups)));axes[0].set_xticklabels(groups,rotation=30,ha="right",fontsize=7)
    axes[0].set_ylabel("Phone-1 AUC on final full-source model");axes[0].legend(fontsize=7)
    selected=metric_table.loc[metric_table.policy=="own_source_loss"]
    base=selected.loc[selected.arm=="full",["group","seed","auc"]]
    for j,arm in enumerate(("apparent_shared","oof_shared")):
        merged=selected.loc[selected.arm==arm].merge(base,on=["group","seed"],suffixes=("","_base"))
        for i,group in enumerate(sites+["EAST_ALL"]):
            part=merged.loc[merged.group==group]
            axes[1].scatter(np.full(len(part),i+(j-.5)*.16),part.auc-part.auc_base,color=f"C{j+1}",label=arm if i==0 else None)
    axes[1].axhline(0,color="black",linewidth=.8);axes[1].set_xticks(range(len(sites)+1));axes[1].set_xticklabels(sites+["EAST_ALL"])
    axes[1].set_ylabel("Whole-population AUC change vs full");axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"figures/03_source_and_whole_site_validation.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=PROJECT/"data/data_mission2.csv")
    parser.add_argument("--previous",type=Path,default=ROOT/"outputs/analysis_2611/task2_step15_east_education_residual_control")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/analysis_2611/task2_step16_east_education_crossfit_check")
    parser.add_argument("--shuffle-repeats",type=int,default=50)
    parser.add_argument("--bootstrap",type=int,default=500)
    args=parser.parse_args()
    if args.shuffle_repeats<50 or args.bootstrap<200:
        raise ValueError("教育打乱至少50次，配对重采样至少200次")
    versions={"python":platform.python_version(),"torch":torch.__version__,"sklearn":sklearn.__version__}
    old=json.loads((args.previous/"experiment_metadata.json").read_text(encoding="utf-8"))
    if not old["completed"] or not old["models_reproduced"] or old["versions"]!=versions:
        raise ValueError("请使用第15步相同环境及完整结果")
    if __import__("scipy").__version__ != old["scipy_version"]:
        raise ValueError("请保持第15步的优化器版本")
    code=Path(previous.__file__).read_text(encoding="utf-8")
    hashes={hashlib.sha256(s.encode()).hexdigest() for s in (code,code.replace("\n","\r\n"))}
    if old["code_sha256"] not in hashes or old["input_sha256"]!=EXPECTED_HASH or hashlib.sha256(args.data.read_bytes()).hexdigest()!=EXPECTED_HASH:
        raise ValueError("第15步代码或固定CSV版本不一致")
    frames=held_out_split(attach_regions(read_raw(args.data)),"east",seed=42,column=REGION)
    if any(set(frames[a].sample_id)&set(frames[b].sample_id) for a,b in ((0,1),(0,2),(1,2))):
        raise RuntimeError("训练、验证、东部存在重叠")
    splits,processor=prepare_splits(frames,environment_column=REGION)
    imputed=[]
    for f in frames:
        part=f.copy();part.loc[:,list(FEATURES)]=processor.imputer.transform(f[list(FEATURES)]);imputed.append(part)
    source,validation,target=imputed
    folds=make_folds(source);spec=design_spec(source)
    designs={arm:[correction_design(f,spec,arm) for f in imputed] for arm in ("phone_intercept","education_shared","education_by_phone")}
    shuffled_designs=[correction_design(shuffle_source(source,r),spec,"education_shared") for r in range(args.shuffle_repeats)]
    old_metrics=pd.read_csv(args.previous/"metrics.csv",dtype={"group":str})
    old_blocks=pd.read_csv(args.previous/"auc_pair_blocks.csv",dtype={"group":str})
    args.output.mkdir(parents=True,exist_ok=False);(args.output/"figures").mkdir()
    meta={"completed":False,"input_sha256":EXPECTED_HASH,"versions":versions,"scipy_version":old["scipy_version"],
          "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seeds":SEEDS,"policies":POLICIES,"arms":ARMS,
          "source_split_seed":42,"folds":5,"fold_seed":FOLD_SEED,"shuffle_repeats":args.shuffle_repeats,
          "shuffle_seed_start":SHUFFLE_SEED,"ridge":previous.RIDGE,"positive_weight":3.,"bootstrap_draws":args.bootstrap,
          "target_used_for_training_or_selection":False,"individual_records_exported":False,
          "primary_comparison":"来源损失选模，OOF共用教育修正vs OOF手机截距，371082与110114手机1；四站及全东部并列",
          "limits":"来源五折模型只用80%来源训练样本，折外分数与完整模型分布可不同；修正应用于原完整MLP，不能把差异全归因训练内过拟合。原来源验证集供完整与各折选模，不是新的独立验证，东部也已反复探索。手机组内打乱保留站点分布及其关联，并非完全无信息；50次分布不是患者置信区间或严格置换P值。配对区间固定三次评分，不包括重新训练不确定性。修正表使用完整来源特征的类别/填补规则，不读取留出标签来预处理；折内MLP预处理严格只读该折训练特征。教育不是已证实因果因素。"}
    dump(args.output/"experiment_metadata.json",meta)
    cohorts=[("SOURCE_TRAIN",0,np.ones(len(source),dtype=bool)),("SOURCE_VALIDATION",1,np.ones(len(validation),dtype=bool)),
             ("EAST_ALL",2,np.ones(len(target),dtype=bool))]
    cohorts += [(str(site),2,target[ENVIRONMENT].to_numpy()==site) for site in sorted(target[ENVIRONMENT].unique())]
    metric_rows=[];block_rows=[];shuffle_metrics=[];shuffle_blocks=[];audit_rows=[];fold_audits=[];oof_probes=[];parameters=[];optimizer_rows=[]
    scores={}
    for seed in SEEDS:
        print(f"任务2 第16步：复现完整MLP，随机种子={seed}",flush=True)
        original=fit_scores(splits,seed,args.output/str(seed)/"full")
        # 在任何五折训练前审计原完整MLP，避免错误基线继续运行。
        for policy in POLICIES:
            for name,index,keep in cohorts:
                row,_=aggregate_scores(imputed[index].loc[keep],original[policy][index][keep],{})
                ref=old_metrics.loc[(old_metrics.arm=="full")&(old_metrics.policy==policy)&(old_metrics.seed==seed)&(old_metrics.group==name)]
                if len(ref)!=1 or any(abs(row[k]-ref.iloc[0][k])>2e-6 for k in ("auc","auprc","brier")):
                    raise RuntimeError("第15步完整模型未复现")
        oof={policy:np.full(len(source),np.nan) for policy in POLICIES}
        seen=np.zeros(len(source),dtype=int)
        for fold,(train,held) in enumerate(folds):
            print(f"任务2 第16步：随机种子={seed}，来源五折={fold+1}/5",flush=True)
            fold_frames=(frames[0].iloc[train],frames[1],frames[0].iloc[held])
            fold_splits,_=prepare_splits(fold_frames,environment_column=REGION)
            if set(fold_splits[2].sample_ids)&(set(fold_splits[0].sample_ids)|set(fold_splits[1].sample_ids)):
                raise RuntimeError("折外样本参与训练或来源选模")
            predictions=fit_scores(fold_splits,seed,args.output/str(seed)/f"fold_{fold+1}")
            seen[held]+=1
            for policy in POLICIES:
                oof[policy][held]=predictions[policy][2]
                for scope,fr in (("FOLD_TRAIN",fold_frames[0]),("OOF_HELDOUT",fold_frames[2])):
                    for phone in (1,2):
                        part=fr.loc[fr[PHONE[0]]==phone]
                        fold_audits.append({"seed":seed,"policy":policy,"fold":fold+1,"scope":scope,"phone":phone,
                              "n":len(part),"positive_n":int(part.label.sum()),"heldout_disjoint_from_train_and_selection":True})
        if not np.all(seen==1) or any(not np.isfinite(p).all() for p in oof.values()):
            raise RuntimeError("五折预测覆盖不完整")
        for policy in POLICIES:
            for phone in (1,2):
                keep=source[PHONE[0]].to_numpy()==phone
                for kind,p in (("apparent",original[policy][0]),("oof",oof[policy])):
                    oof_probes.append({"seed":seed,"policy":policy,"scope":kind,"phone":phone,
                        **metrics(source.loc[keep,"label"].to_numpy(dtype=int),p[keep])})
            for arm in ARMS:
                if arm=="full":
                    predictions=original[policy]
                else:
                    kind="education_by_phone" if arm=="oof_by_phone" else "education_shared" if arm.endswith("shared") else "phone_intercept"
                    fitting_p=oof[policy] if arm.startswith("oof") else original[policy][0]
                    beta,opt=fit_correction(designs[kind][0],logit(np.clip(fitting_p,1e-12,1-1e-12)),source.label.to_numpy(dtype=int))
                    parameters.extend({"seed":seed,"policy":policy,"arm":arm,"term_index":i,"coefficient":float(v)} for i,v in enumerate(beta))
                    optimizer_rows.append({"seed":seed,"policy":policy,"arm":arm,"repeat":-1,**opt})
                    predictions=[apply_correction(p,d,beta) for p,d in zip(original[policy],designs[kind])]
                for name,index,keep in cohorts:
                    context={"seed":seed,"policy":policy,"arm":arm,"group":name}
                    p=predictions[index][keep];scores[seed,policy,arm,name]=p
                    row,blocks=aggregate_scores(imputed[index].loc[keep],p,context)
                    metric_rows.append(row);block_rows.extend(blocks)
                    if arm in ("full","apparent_shared"):
                        old_arm="full" if arm=="full" else "education_shared"
                        ref=old_metrics.loc[(old_metrics.arm==old_arm)&(old_metrics.seed==seed)&(old_metrics.policy==policy)&(old_metrics.group==name)]
                        ref_b=old_blocks.loc[(old_blocks.arm==old_arm)&(old_blocks.seed==seed)&(old_blocks.policy==policy)&(old_blocks.group==name)]
                        passed=len(ref)==1 and all(abs(row[k]-ref.iloc[0][k])<=2e-6 for k in ("auc","auprc","brier"))
                        for b in blocks:
                            prior=ref_b.loc[ref_b.block==b["block"]]
                            passed=passed and len(prior)==1 and np.isclose(b["block_auc"],prior.iloc[0].block_auc,equal_nan=True,atol=2e-6,rtol=0)
                        audit_rows.append({**context,"passed":bool(passed)})
                        if not passed:raise RuntimeError("原模型或第15步共用教育修正未复现")
                    if arm in ("apparent_phone","oof_phone"):
                        before,_=block_metrics(imputed[index].loc[keep],original[policy][index][keep])
                        for a,b in zip(before,blocks):
                            if a["block"] in ("P1_N1","P2_N2") and not np.isclose(a["block_auc"],b["block_auc"],equal_nan=True,atol=1e-10,rtol=0):
                                raise RuntimeError("手机截距改变了组内排序")
            for repeat,design in enumerate(shuffled_designs):
                if repeat%10==0:print(f"任务2 第16步：{policy}，重复教育打乱={repeat+1}/{args.shuffle_repeats}",flush=True)
                beta,opt=fit_correction(design,logit(np.clip(oof[policy],1e-12,1-1e-12)),source.label.to_numpy(dtype=int))
                optimizer_rows.append({"seed":seed,"policy":policy,"arm":"oof_shared_shuffled","repeat":repeat,**opt})
                predictions=[apply_correction(p,d,beta) for p,d in zip(original[policy],designs["education_shared"])]
                for name,index,keep in cohorts:
                    if name=="SOURCE_TRAIN":continue
                    context={"seed":seed,"policy":policy,"group":name,"repeat":repeat}
                    row,blocks=aggregate_scores(imputed[index].loc[keep],predictions[index][keep],context)
                    shuffle_metrics.append(row);shuffle_blocks.extend(blocks)
    metric_table=pd.DataFrame(metric_rows);blocks=pd.DataFrame(block_rows);interval_rows=[]
    for name,index,keep in cohorts:
        if name=="SOURCE_TRAIN":continue
        for policy in POLICIES:
            for arm,ref in CONTRASTS:
                before=np.stack([scores[s,policy,ref,name] for s in SEEDS]);after=np.stack([scores[s,policy,arm,name] for s in SEEDS])
                rows=bootstrap_change(imputed[index].loc[keep],before,after,args.bootstrap,seed=FOLD_SEED)
                interval_rows.extend({"group":name,"policy":policy,"arm":arm,"reference":ref,**r} for r in rows)
    intervals=pd.DataFrame(interval_rows);shuffle_blocks=pd.DataFrame(shuffle_blocks)
    summary=shuffle_summary(blocks,shuffle_blocks)
    tables={"metrics":metric_table,"auc_pair_blocks":blocks,"paired_intervals":intervals,
            "shuffle_metrics":pd.DataFrame(shuffle_metrics),"shuffle_auc_pair_blocks":shuffle_blocks,"shuffle_summary":summary,
            "model_reproduction_audits":pd.DataFrame(audit_rows),"fold_disjointness_audits":pd.DataFrame(fold_audits),
            "source_oof_vs_apparent":pd.DataFrame(oof_probes),"correction_coefficients":pd.DataFrame(parameters),
            "optimizer_audits":pd.DataFrame(optimizer_rows)}
    for name,table in tables.items():table.to_csv(args.output/(name+".csv"),index=False,encoding="utf-8-sig")
    dump(args.output/"design_protocol.json",{"categories":[float(v) for v in spec["categories"]],
         "global_center":spec["global_mean"].tolist(),"group_centers":{str(k):v.tolist() for k,v in spec["group_mean"].items()},
         "folds":5,"correction_fitting":"来源训练标签加对应样本折外预测；来源验证与东部不拟合修正表"})
    draw(blocks,intervals,summary,metric_table,args.output)
    meta.update(completed=True,models_reproduced=True,oof_coverage_complete=True,heldout_not_used_for_model_selection=True)
    dump(args.output/"experiment_metadata.json",meta)
    print(f"任务2 第16步完成，输出目录：{args.output}")


if __name__=="__main__":
    main()
