"""任务3 第8步：稀疏不变学习的低维语义特征适配试验。

依据UAI 2025 Du与Banerjee论文公式11、算法1：环境风险加梯度平方惩罚，
梯度更新后做硬阈值投影。二分类加权BCE、截距及类别整组投影是本项目适配，
不是官方基准复现，不承诺因果识别。四对照共用预处理、损失、环境权重和优化器。
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, read_raw, attach_regions, held_out_split

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
METHODS = ("erm", "erm_iht", "irm", "irm_iht")
ENCODINGS = ("numeric", "categorical_grouped")
SEEDS = (42, 43, 44)
BUDGETS = (12,)
PENALTIES = (.1, 1.)
RIDGE = 1e-4
POSITIVE_WEIGHT = 3.
PAPER = "https://proceedings.mlr.press/v286/du25b.html"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fit_basis(source, encoding):
    """任务3 第8步：只用来源训练特征拟合填补、类别集合、中心及尺度。"""
    imputer = SimpleImputer(strategy="median")
    raw = imputer.fit_transform(source[list(FEATURES)])
    if raw.shape[1] != len(FEATURES):
        raise ValueError("来源存在整列缺失，禁止静默删列")
    basis = {"encoding": encoding, "imputer": imputer, "levels": {}, "groups": [], "names": []}
    matrices = []
    for i, feature in enumerate(FEATURES):
        if encoding == "numeric":
            values = raw[:, i:i+1]
            labels = [feature]
        elif encoding == "categorical_grouped":
            levels = np.unique(raw[:, i]); basis["levels"][feature] = levels
            values = (raw[:, i, None] == levels[None, :]).astype(float)
            labels = [f"{feature}={v:g}" for v in levels]
        else:
            raise ValueError("未知编码")
        mean = values.mean(axis=0); scale = values.std(axis=0)
        valid = scale > 1e-12
        start = sum(m.shape[1] for m in matrices)
        matrices.append((values[:, valid]-mean[valid])/scale[valid])
        basis["groups"].append(np.arange(start, start+int(valid.sum())))
        basis["names"].extend([label for label, keep in zip(labels, valid) if keep])
        basis[feature] = (mean, scale, valid)
    x = np.column_stack(matrices)
    return basis, np.column_stack([x, np.ones(len(x))])


def transform_basis(frame, basis):
    """任务3 第8步：未知类别的整组贡献设为零；不扩展目标地区出现的新类别。"""
    raw = basis["imputer"].transform(frame[list(FEATURES)])
    matrices, unknown = [], []
    for i, feature in enumerate(FEATURES):
        if basis["encoding"] == "numeric":
            values = raw[:, i:i+1]; known = np.ones(len(raw), dtype=bool)
        else:
            levels = basis["levels"][feature]
            values = (raw[:, i, None] == levels[None, :]).astype(float)
            known = values.sum(axis=1) > 0
        mean, scale, valid = basis[feature]
        standardized = (values[:, valid]-mean[valid])/scale[valid]
        standardized[~known] = 0
        matrices.append(standardized)
        unknown.append({"feature": feature, "unknown_n": int((~known).sum())})
    return np.column_stack(matrices+[np.ones((len(raw), 1))]), unknown


def project_groups(weights, groups, budget):
    """任务3 第8步：保留L2范数最大的若干原特征组；截距不投影，数值编码即坐标IHT。"""
    if budget is None:
        return weights.copy()
    norms = np.array([np.linalg.norm(weights[index]) for index in groups])
    order = np.argsort(-norms, kind="stable")
    keep = set(order[:budget].tolist())
    result = weights.copy()
    for i, index in enumerate(groups):
        if i not in keep:
            result[index] = 0
    return result


def environment_arrays(x, y, sites):
    """任务3 第8步：来源县区作为环境；所有四对照按县区等权，而非按人数加权。"""
    return [(x[sites == site], y[sites == site]) for site in sorted(set(sites))]


def objective(weights, environments, rho):
    """任务3 第8步：梯度平方惩罚覆盖完整斜率和截距，含解析二阶链式导数。"""
    loss, penalty = 0., 0.
    gradient = np.zeros_like(weights)
    for x, y in environments:
        z = x@weights; p = expit(z)
        sample_weight = np.where(y == 1, POSITIVE_WEIGHT, 1.)
        risk = float(np.mean(sample_weight*(np.logaddexp(0, z)-y*z)))
        g = x.T@(sample_weight*(p-y))/len(y)
        loss += risk; penalty += float(g@g)
        gradient += g
        if rho:
            hessian_g = x.T@(sample_weight*p*(1-p)*(x@g))/len(y)
            gradient += 2*rho*hessian_g
    count = len(environments)
    regularized = weights.copy(); regularized[-1] = 0
    total = (loss+rho*penalty)/count+.5*RIDGE*float(regularized@regularized)
    return total, gradient/count+RIDGE*regularized, loss/count, penalty/count


def validation_score(y, p, sites):
    """任务3 第8步：按来源验证县区宏AUC选轮次及超参；不读取留出地区标签。"""
    aucs = [roc_auc_score(y[sites == s], p[sites == s]) for s in sorted(set(sites))
            if len(np.unique(y[sites == s])) == 2]
    if not aucs:
        raise ValueError("来源验证没有可计算AUC的站点")
    loss = float(np.mean(np.where(y == 1, POSITIVE_WEIGHT, 1.)*
                         (-y*np.log(np.clip(p,1e-12,1))-(1-y)*np.log(np.clip(1-p,1e-12,1)))))
    return float(np.mean(aucs)), loss, len(aucs)


def train_candidate(x, y, sites, xv, yv, sv, groups, seed, rho, budget, steps):
    """任务3 第8步：全批量投影梯度下降加回溯；投影后检查目标，避免数值发散。"""
    env = environment_arrays(x,y,sites)
    weights = np.random.default_rng(seed).normal(0,.01,x.shape[1]);weights[-1]=0
    weights = project_groups(weights,groups,budget)
    best, best_key, best_epoch = None, None, None
    history = []
    for epoch in range(steps+1):
        value, gradient, risk, penalty = objective(weights,env,rho)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError("训练目标或梯度出现非有限值")
        if epoch%25 == 0 or epoch == steps:
            macro, vl, eligible = validation_score(yv,expit(xv@weights),sv)
            key = (round(macro,12),-vl)
            history.append({"epoch":epoch,"objective":value,"risk":risk,"gradient_penalty":penalty,
                            "source_validation_macro_auc":macro,"source_validation_weighted_bce":vl,
                            "eligible_validation_sites":eligible})
            if epoch > 0 and (best_key is None or key>best_key):
                best_key,best,best_epoch=key,weights.copy(),epoch
        if epoch == steps:
            break
        step=.1
        while True:
            proposal=project_groups(weights-step*gradient,groups,budget)
            next_value=objective(proposal,env,rho)[0]
            if np.isfinite(next_value) and next_value <= value+1e-12:
                weights=proposal
                break
            step*=.5
            if step<1e-12:
                raise RuntimeError("回溯无法找到下降步长，请保留本次结果核查")
    return {"weights":best,"source_macro":best_key[0],"source_loss":-best_key[1],
            "epoch":best_epoch,"rho":rho,"budget":budget,"history":history}


def candidates(method):
    penalties = PENALTIES if method in ("irm","irm_iht") else (0.,)
    budgets = BUDGETS if method in ("erm_iht","irm_iht") else (None,)
    return [(rho,budget) for rho in penalties for budget in budgets]


def aggregate_metrics(y,p):
    """任务3 第8步：排序、概率误差与固定阈值召回分别记录；单类小组不硬算AUC。"""
    both=len(np.unique(y))==2
    return {"auc":float(roc_auc_score(y,p)) if both else np.nan,
            "auprc":float(average_precision_score(y,p)) if y.sum() else np.nan,
            "brier":float(brier_score_loss(y,p)),"recall_at_05":float(np.mean(p[y==1]>.5)) if y.sum() else np.nan,
            "n":len(y),"positive_n":int(y.sum()),"positive_rate":float(y.mean())}


def cohort_masks(frame):
    """任务3 第8步：地区合并、站点、手机组分别展示，禁止把合并AUC冒充站点宏平均。"""
    result=[("REGION_ALL","all",np.ones(len(frame),dtype=bool))]
    result += [(str(s),"all",frame[ENVIRONMENT].to_numpy()==s) for s in sorted(frame[ENVIRONMENT].unique())]
    for name,_,keep in list(result):
        for phone in (1,2):
            result.append((name,f"phone{phone}",keep&(frame["if_use_mobile_phone"].to_numpy()==phone)))
    return [(name,subgroup,keep) for name,subgroup,keep in result if keep.any()]


def education_permutation(frame,x,weights,columns,repeats):
    """任务3 第8步：同站点×手机组内置换教育块；各方法共用次序，不交换其他特征。"""
    groups=frame.groupby([ENVIRONMENT,"if_use_mobile_phone"],sort=True).indices
    baseline=expit(x@weights); rows=[]
    for repeat in range(repeats):
        rng=np.random.default_rng(26113800+repeat);order=np.arange(len(frame))
        for index in groups.values():
            order[index]=rng.permutation(index)
        changed=x.copy();changed[:,columns]=x[order][:,columns]
        p=expit(changed@weights)
        for name,subgroup,keep in cohort_masks(frame):
            y=frame.label.to_numpy(dtype=int)[keep]
            if len(np.unique(y))<2:continue
            original=float(roc_auc_score(y,baseline[keep]));altered=float(roc_auc_score(y,p[keep]))
            rows.append({"group":name,"subgroup":subgroup,"repeat":repeat,"baseline_auc":original,
                         "permuted_auc":altered,"auc_drop":original-altered})
    return rows


def draw(table,coefficients,reliance,output):
    """任务3 第8步：跨地区泛化、语义特征保留、教育依赖三张固定图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    part=table.loc[(table.stage=="test")&(table.group!="REGION_ALL")&(table.subgroup=="all")]
    for ax,encoding in zip(axes,ENCODINGS):
        values=part.loc[part.encoding==encoding].groupby(["held_out","method"]).auc.mean().unstack().reindex(["east","central","west"])
        values.reindex(columns=METHODS).plot.bar(ax=ax,rot=0);ax.set_title(encoding);ax.set_ylabel("Equal-site macro AUC")
    fig.tight_layout();fig.savefig(output/"figures/01_sparse_invariance_region_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
    for ax,encoding in zip(axes,ENCODINGS):
        values=coefficients.loc[(coefficients.encoding==encoding)&(coefficients.held_out=="east")].groupby(["feature","method"]).selected.mean().unstack().reindex(index=list(FEATURES),columns=METHODS)
        im=ax.imshow(values,aspect="auto",vmin=0,vmax=1,cmap="Blues");ax.set_title(f"{encoding}: East held out")
        ax.set_xticks(range(len(METHODS)));ax.set_xticklabels(METHODS,rotation=20);ax.set_yticks(range(len(FEATURES)));ax.set_yticklabels(FEATURES,fontsize=8)
    fig.subplots_adjust(left=.17,right=.85,wspace=.15,bottom=.12,top=.93)
    color_axis=fig.add_axes([.89,.2,.015,.6])
    fig.colorbar(im,cax=color_axis,label="Selection frequency across three seeds")
    fig.savefig(output/"figures/02_semantic_feature_retention.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,site in zip(axes,("110114","371082")):
        for j,encoding in enumerate(ENCODINGS):
            values=reliance.loc[(reliance.held_out=="east")&(reliance.stage=="test")&(reliance.group==site)&(reliance.subgroup=="phone1")&(reliance.encoding==encoding)].groupby("method").auc_drop.mean().reindex(METHODS)
            ax.bar(np.arange(len(METHODS))+(j-.5)*.3,values,width=.3,label=encoding)
        ax.axhline(0,color="black",linewidth=.8);ax.set_xticks(range(len(METHODS)));ax.set_xticklabels(METHODS,rotation=20)
        ax.set_title(site+": phone-1");ax.set_ylabel("Education permutation AUC drop");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"figures/03_education_information_preservation.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=PROJECT/"data/data_mission2.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/analysis_2611/task3_step8_sparse_invariance_feature_pilot")
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--permutations",type=int,default=20)
    args=parser.parse_args()
    if args.steps<500 or args.permutations<10:raise ValueError("至少训练500轮，教育置换至少10次")
    if hashlib.sha256(args.data.read_bytes()).hexdigest()!=EXPECTED_HASH:raise ValueError("固定CSV版本不一致")
    args.output.mkdir(parents=True,exist_ok=False);(args.output/"figures").mkdir()
    metadata={"completed":False,"input_sha256":EXPECTED_HASH,"paper":PAPER,"paper_year":2025,
              "paper_equations":"公式11和算法1的梯度平方惩罚及IHT；未实现公式12内层极小极大求解",
              "adaptations":"固定原始语义输入代替预训练表示；二分类正类权重3；截距纳入环境梯度惩罚但不稀疏化；类别编码整组投影；环境风险取等权均值；固定L2与回溯步长",
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
              "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seeds":SEEDS,"split_seed":42,
              "source_environments":"county/site","environment_weighting":"equal_site_in_all_four_arms",
              "positive_weight":POSITIVE_WEIGHT,"ridge":RIDGE,"budgets":BUDGETS,"penalties":PENALTIES,
              "steps":args.steps,"permutations":args.permutations,"encodings":ENCODINGS,"methods":METHODS,
              "target_used_for_training_or_selection":False,"individual_records_exported":False,
              "limits":"这是低维分类适配，不是官方图像基准复现或因果识别。两种编码仅在各自内部做四对照，不能把编码差异归因不变性。类别组维数不同可影响组范数排名。来源验证反复选择超参与轮次会乐观；目标地区已反复探索。环境患病率等差异可使不变约束过强。教育留存或置换下降均不是因果效应；无隐层，本实验不检验隐层表征崩溃，只记录输出分数退化。与历史MLP的结构和环境等权损失不同，不作单因素结构归因。"}
    dump(args.output/"experiment_metadata.json",metadata)
    frame=attach_regions(read_raw(args.data))
    metrics_rows=[];feature_rows=[];selection_rows=[];permutation_rows=[];diagnostics=[];encoding_audits=[];unknown_rows=[]
    for region in ("east","central","west"):
        train,val,target=held_out_split(frame,region,seed=42,column=REGION)
        if any(set(a.sample_id)&set(b.sample_id) for a,b in ((train,val),(train,target),(val,target))):
            raise RuntimeError("来源训练、验证、留出地区重叠")
        for encoding in ENCODINGS:
            basis,x=fit_basis(train,encoding);xv,uv=transform_basis(val,basis);xt,ut=transform_basis(target,basis)
            for stage,unknown in (("source_validation",uv),("test",ut)):
                unknown_rows.extend({"held_out":region,"encoding":encoding,"stage":stage,**r} for r in unknown)
            basis_folder=args.output/region/encoding;basis_folder.mkdir(parents=True,exist_ok=True)
            dump(basis_folder/"feature_encoding.json", {"encoding":encoding,"features":list(FEATURES),
                 "imputation_medians":basis["imputer"].statistics_.tolist(),"column_names":basis["names"],
                 "levels":{k:v.tolist() for k,v in basis["levels"].items()},
                 "centers_scales_valid":{f:[v.tolist() for v in basis[f]] for f in FEATURES},
                 "group_indices":[v.tolist() for v in basis["groups"]]})
            encoding_audits.append({"held_out":region,"encoding":encoding,"input_columns":x.shape[1]-1,
                   "semantic_groups":len(basis["groups"]),"train_n":len(train),"validation_n":len(val),"target_n":len(target),"disjoint":True})
            y=train.label.to_numpy(dtype=int);yv=val.label.to_numpy(dtype=int)
            sites=train[ENVIRONMENT].to_numpy();sv=val[ENVIRONMENT].to_numpy()
            for seed in SEEDS:
                selected={}
                for method in METHODS:
                    best=None;best_key=None
                    for rho,budget in candidates(method):
                        print(f"任务3 第8步：留出{region}，编码={encoding}，种子={seed}，方法={method}，约束={rho}/{budget}",flush=True)
                        result=train_candidate(x,y,sites,xv,yv,sv,basis["groups"],seed,rho,budget,args.steps)
                        selection_rows.append({"held_out":region,"encoding":encoding,"seed":seed,"method":method,
                            "rho":rho,"budget":budget,"epoch":result["epoch"],"source_macro_auc":result["source_macro"],"source_weighted_bce":result["source_loss"]})
                        key=(result["source_macro"],-result["source_loss"],-rho,-(budget or len(FEATURES)))
                        if best_key is None or key>best_key:best_key,best=key,result
                    selected[method]=best
                # 所有条件完成来源选模后，才评估留出地区；不按目标表现重新选条件。
                for method,result in selected.items():
                    context={"held_out":region,"encoding":encoding,"seed":seed,"method":method}
                    weights=result["weights"]
                    folder=args.output/region/encoding/str(seed)/method;folder.mkdir(parents=True)
                    np.savez(folder/"selected_parameters.npz",weights=weights,column_names=np.array(basis["names"]+["intercept"]))
                    dump(folder/"source_selection.json",{k:v for k,v in result.items() if k!="weights"})
                    for feature,index in zip(FEATURES,basis["groups"]):
                        norm=float(np.linalg.norm(weights[index]))
                        feature_rows.append({**context,"feature":feature,"group_norm":norm,"selected":norm>1e-8,"columns":len(index)})
                    active=sum(np.linalg.norm(weights[ix])>1e-8 for ix in basis["groups"])
                    if result["budget"] is not None and active>result["budget"]:raise RuntimeError("硬阈值未保持特征预算")
                    for stage,part,values in (("source_train",train,x),("source_validation",val,xv),("test",target,xt)):
                        p=expit(values@weights)
                        for group,subgroup,keep in cohort_masks(part):
                            metrics_rows.append({**context,"stage":stage,"group":group,"subgroup":subgroup,**aggregate_metrics(part.label.to_numpy(dtype=int)[keep],p[keep])})
                        diagnostics.append({**context,"stage":stage,"selected_groups":int(active),"score_std":float(p.std()),
                            "score_collapsed":bool(p.std()<1e-6),"education_selected":bool(np.linalg.norm(weights[basis["groups"][FEATURES.index("education")]])>1e-8)})
                        if stage!="source_train":
                            rows=education_permutation(part,values,weights,basis["groups"][FEATURES.index("education")],args.permutations)
                            permutation_rows.extend({**context,"stage":stage,**r} for r in rows)
                # 每个来源种子完成后保存聚合检查点，长任务中断时仍可核查已完成部分。
                pd.DataFrame(selection_rows).to_csv(args.output/"source_only_selection.csv",index=False,encoding="utf-8-sig")
                pd.DataFrame(metrics_rows).to_csv(args.output/"metrics.csv",index=False,encoding="utf-8-sig")
    table=pd.DataFrame(metrics_rows);features=pd.DataFrame(feature_rows);reliance=pd.DataFrame(permutation_rows)
    site_test=table.loc[(table.stage=="test")&(table.group!="REGION_ALL")&(table.subgroup=="all")]
    macro=site_test.groupby(["held_out","encoding","method","seed"],as_index=False)[["auc","auprc","brier"]].mean()
    tables={"semantic_feature_support":features,"education_permutation_repeats":reliance,
            "prediction_diagnostics":pd.DataFrame(diagnostics),"encoding_audits":pd.DataFrame(encoding_audits),
            "unknown_categories":pd.DataFrame(unknown_rows),"region_site_macro":macro,
            "global_site_macro":site_test.groupby(["encoding","method","seed"],as_index=False)[["auc","auprc","brier"]].mean()}
    for name,values in tables.items():values.to_csv(args.output/(name+".csv"),index=False,encoding="utf-8-sig")
    draw(table,features,reliance,args.output)
    metadata.update(completed=True)
    dump(args.output/"experiment_metadata.json",metadata)
    print(f"任务3 第8步完成，输出目录：{args.output}")


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
