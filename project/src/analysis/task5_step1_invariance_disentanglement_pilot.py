"""任务五 第1步：ICLR 2025不变性解耦的低维机制复现，由用户运行。

依据：Yao et al., Unifying Causal Representation Learning with the Invariance
Principle，正式版本§5.2、附录F.2。复现SCM参数和“重建+MMD”原则。
这是CPU缩小规模的独立实现，不是作者代码或原文数字的精确复现。
不读取老年健康数据；训练时不使用真实潜变量，真实变量只用于训练后诊断。
运行：python project/src/analysis/task5_step1_invariance_disentanglement_pilot.py
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/analysis_2611/task5_step1_invariance_disentanglement_pilot"
PAPER = "https://proceedings.iclr.cc/paper_files/paper/2025/file/85381f4549b5ddf1d48e2e287d7d3d15-Paper-Conference.pdf"
ARMS = ("reconstruction_only", "reconstruction_plus_mmd")
SEED = 42


def latent_sample(n, seed, shifted_mean=None):
    """任务五 第1步：严格使用F.1参数；替换z2后，z3仍由旧z2产生。"""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(10.5, 0.8, n)
    old_z2 = 0.02*z1 + rng.normal(0, 0.5, n)
    z3 = old_z2 + 3 + rng.normal(0, 1, n)
    z2 = old_z2 if shifted_mean is None else rng.normal(shifted_mean, 0.02, n)
    return np.column_stack([z1, z2, z3])


def make_mixing(seed):
    """任务五 第1步：三层可逆混合，固定正交方阵，保证不会预先丢失变量。"""
    rng = np.random.default_rng(seed)
    return [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(3)]


def mix(z, matrices):
    """任务五 第1步：混合在训练与评估中保持相同；与原文隐藏宽度不同，显式记录。"""
    # 任务五 第1步：常数缩放只为数值稳定，不改变真实变量的信息。
    x = (np.asarray(z) - np.array([10.5, 0.0, 3.0])) / np.array([0.8, 1.0, 1.0])
    for i, matrix in enumerate(matrices):
        x = x @ matrix
        if i < 2:
            x = np.where(x >= 0, x, 0.2*x)
    return x


def make_data(n, test_n, trial, shifted_mean):
    """任务五 第1步：独立随机流生成训练与评估数据；预处理只拟合训练集。"""
    matrices = make_mixing(SEED + 500)
    train_z = [latent_sample(n, SEED+1000*trial+i, mu)
               for i, mu in enumerate((None, shifted_mean))]
    test_z = [latent_sample(test_n, SEED+1000*trial+100+i, mu)
              for i, mu in enumerate((None, shifted_mean))]
    scaler = StandardScaler().fit(np.vstack([mix(z, matrices) for z in train_z]))
    train_x = [scaler.transform(mix(z, matrices)).astype(np.float32) for z in train_z]
    test_x = [scaler.transform(mix(z, matrices)).astype(np.float32) for z in test_z]
    return train_x, test_x, train_z, test_z, matrices, scaler


def gaussian_mmd(left, right):
    """任务五 第1步：固定多尺度高斯核、有偏MMD平方估计，不裁剪梯度。"""
    import torch
    def kernel(a, b):
        distances = torch.cdist(a, b).square()
        return sum(torch.exp(-gamma*distances) for gamma in (0.5, 1.0, 2.0))/3
    return kernel(left, left).mean() + kernel(right, right).mean() - 2*kernel(left, right).mean()


def build_autoencoder():
    """任务五 第1步：编码器与解码器均为三层MLP，隐藏宽度128、潜空间3维。"""
    from torch import nn
    def network():
        return nn.Sequential(nn.Linear(3, 128), nn.LeakyReLU(0.2),
                             nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, 3))
    return nn.ModuleDict({"encoder": network(), "decoder": network()})


def fit_model(train_x, arm, epochs, batch_size, seed):
    """任务五 第1步：只接收观测x，不接收真实z；两组初始化与批次相同。"""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = build_autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    generator = torch.Generator().manual_seed(seed+1)
    domains = [torch.as_tensor(x) for x in train_x]
    history = []
    weight = float(arm == ARMS[1])
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    for epoch in range(1, epochs+1):
        totals = np.zeros(3)
        steps = math.ceil(len(domains[0])/batch_size)
        # 任务五 第1步：两个环境独立打乱，训练不使用跨环境逐人配对。
        orders = [torch.randperm(len(x), generator=generator) for x in domains]
        model.train()
        for start in range(0, len(domains[0]), batch_size):
            batches = [x[order[start:start+batch_size]] for x, order in zip(domains, orders)]
            encoded = [model["encoder"](x) for x in batches]
            reconstruction = sum((model["decoder"](h)-x).square().sum(1).mean()
                                 for x, h in zip(batches, encoded))
            penalty = gaussian_mmd(encoded[0][:, [0, 2]], encoded[1][:, [0, 2]])
            objective = reconstruction + weight*penalty
            if not torch.isfinite(objective):
                raise FloatingPointError(f"非有限损失：{arm}, epoch={epoch}")
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            totals += [objective.item(), reconstruction.item(), penalty.item()]
        history.append({"epoch": epoch, "objective": totals[0]/steps,
                        "reconstruction": totals[1]/steps, "mmd_squared": totals[2]/steps})
        if epoch == 1 or epoch % 25 == 0:
            print(f"{arm}: epoch={epoch}, reconstruction={totals[1]/steps:.5f}, MMD={totals[2]/steps:.5f}", flush=True)
    model.eval()
    return model, history, initial_hash


def encode(model, x):
    """任务五 第1步：冻结模型，批量提取表征与重建数据。"""
    import torch
    with torch.no_grad():
        latent = model["encoder"](torch.as_tensor(x))
        reconstructed = model["decoder"](latent)
    return latent.numpy(), reconstructed.numpy()


def probe(train_h, test_h, train_z, test_z, info, representation):
    """任务五 第1步：训练后拟合固定KRR读出器，独立测试集评估真实潜变量信息。"""
    # 任务五 第1步：读出器用到z，但不会更新自编码器、选训练轮次或选MMD权重。
    rng = np.random.default_rng(SEED+900)
    ids = rng.choice(len(train_h), min(1024, len(train_h)), replace=False)
    scaler = StandardScaler().fit(train_h[ids])
    model = KernelRidge(alpha=0.01, kernel="rbf", gamma=1/train_h.shape[1])
    target_scaler = StandardScaler().fit(train_z[ids])
    model.fit(scaler.transform(train_h[ids]), target_scaler.transform(train_z[ids]))
    predicted = target_scaler.inverse_transform(model.predict(scaler.transform(test_h)))
    return [{**info, "representation": representation, "latent": f"z{i+1}",
             "test_r2": float(r2_score(test_z[:, i], predicted[:, i])),
             "probe_fit_n": len(ids), "probe_test_n": len(test_h)} for i in range(3)]


def diagnose(model, train_x, test_x, train_z, test_z, info):
    """任务五 第1步：同时检查恢复能力、重建误差及表征秩，防止把坍缩当解耦。"""
    train_h = np.vstack([encode(model, x)[0] for x in train_x])
    h_and_recon = [encode(model, x) for x in test_x]
    test_h = np.vstack([pair[0] for pair in h_and_recon])
    z_train, z_test = np.vstack(train_z), np.vstack(test_z)
    # 任务五 第1步：补查标准化后的分布差异和环境可分性，避免仅靠缩小表征骗过MMD。
    import torch
    block_scaler = StandardScaler().fit(train_h[:, [0, 2]])
    blocks = [h[:, [0, 2]] for h, _ in h_and_recon]
    with torch.no_grad():
        raw_mmd = float(gaussian_mmd(*(torch.as_tensor(b[:512].copy()) for b in blocks)))
        scaled_mmd = float(gaussian_mmd(*(torch.as_tensor(block_scaler.transform(b[:512]).astype(np.float32)) for b in blocks)))
    domain_train = np.repeat([0, 1], [len(x) for x in train_x])
    domain_test = np.repeat([0, 1], [len(x) for x in test_x])
    domain_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    domain_probe.fit(block_scaler.transform(train_h[:, [0, 2]]), domain_train)
    domain_auc = float(roc_auc_score(domain_test, domain_probe.predict_proba(block_scaler.transform(test_h[:, [0, 2]]))[:, 1]))
    rows = []
    for name, a, b in (("invariant_block_1_3", train_h[:, [0, 2]], test_h[:, [0, 2]]),
                       ("full_representation", train_h, test_h),
                       ("observed_x_reference", np.vstack(train_x), np.vstack(test_x)),
                       ("true_z1_z3_reference", z_train[:, [0, 2]], z_test[:, [0, 2]])):
        rows.extend(probe(a, b, z_train, z_test, info, name))
    diagnostics = []
    for domain, x, (h, reconstruction) in zip(("observational", "nintervention"), test_x, h_and_recon):
        centered = h-h.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        diagnostics.append({**info, "domain": domain,
                            "reconstruction_mse": float(np.mean((x-reconstruction)**2)),
                            "latent_std_1": float(h[:, 0].std()), "latent_std_2": float(h[:, 1].std()),
                            "latent_std_3": float(h[:, 2].std()),
                            "latent_effective_rank": int(np.linalg.matrix_rank(centered)),
                            "smallest_singular_value": float(singular[-1]),
                            "heldout_block_mmd_raw": raw_mmd,
                            "heldout_block_mmd_standardized": scaled_mmd,
                            "heldout_linear_domain_probe_auc": domain_auc})
    return rows, diagnostics


def draw_figures(recovery, diagnostics, history, output):
    """任务五 第1步：只画模拟结果，不生成研究结论。"""
    figures = output / "figures"
    figures.mkdir()
    part = recovery.loc[recovery.representation.eq("invariant_block_1_3")]
    fig, ax = plt.subplots(figsize=(8, 5))
    means = part.groupby(["latent", "arm"]).test_r2.mean().unstack("arm").reindex(columns=ARMS)
    means.plot.bar(ax=ax, rot=0)
    for j, arm in enumerate(ARMS):
        for i, latent in enumerate(means.index):
            values = part.loc[part.arm.eq(arm) & part.latent.eq(latent), "test_r2"]
            ax.scatter(np.full(len(values), i+(j-0.5)*0.25), values, color="black", s=18, zorder=4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Held-out R2 from learned block [h1, h3]")
    ax.set_title("Downscaled mechanism reproduction; dots = three trials")
    fig.tight_layout()
    fig.savefig(figures / "invariant_block_latent_recovery.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in ARMS:
        part = history.loc[history.arm.eq(arm)].groupby("epoch")[["reconstruction", "mmd_squared"]].mean()
        axes[0].plot(part.index, part.reconstruction, label=arm)
        axes[1].plot(part.index, part.mmd_squared, label=arm)
    axes[0].set_ylabel("Training reconstruction sum (two environments)")
    axes[1].set_ylabel("Training MMD squared")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "reconstruction_and_alignment.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"禁止覆盖已有结果：{args.output}")
    import torch
    samples, test_n, epochs, batch_size = 6000, 2000, 100, 256
    means = np.random.default_rng(SEED).uniform(2, 5, size=3)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"task": "Task 5 Step 1: ICLR2025 Section5.2 mechanism pilot",
                "paper": PAPER, "reproduction_level": "independent_downscaled_mechanism_reproduction",
                "health_data_used": False, "latent_truth_used_for_autoencoder_training": False,
                "checkpoint_selection": "fixed final epoch; no oracle latent selection",
                "samples_per_environment": samples, "test_samples_per_environment": test_n,
                "epochs": epochs, "batch_size_per_environment": batch_size,
                "mmd_weight": 1.0, "mmd_kernel_gammas": [0.5, 1.0, 2.0],
                "seed": SEED, "nintervention_means": means.tolist(),
                "torch_version": torch.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "paper_training_budget": {"samples": 200000, "epochs": 500, "batch_size": 4000},
                "deviations": ["Reduced CPU training budget; not a replication of published numerical scores.",
                               "Mixing: three invertible 3x3 layers instead of unspecified random wide invertible MLP.",
                               "Training-only observed-data standardization; fixed LeakyReLU slope 0.2.",
                               "Explicit fixed Gaussian kernel mixture and fixed KRR settings; these details are not fully specified in the paper.",
                               "Added reconstruction-only baseline and independent probe evaluation; no parameter search."],
                "limitations": ["Known continuous synthetic SCM, not discrete health features.",
                                "Invariance block indices are supplied, not discovered from site labels.",
                                "Block recovery is not individual factor identification or causal-graph recovery.",
                                "MMD reduction alone is not successful disentanglement; check information preservation.",
                                "Three shift trials are descriptive and do not validate general superiority."]}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows, all_diagnostics, all_history, checks = [], [], [], []
    for trial, mu in enumerate(means):
        train_x, test_x, train_z, test_z, matrices, scaler = make_data(samples, test_n, trial, float(mu))
        np.savez(args.output / f"trial_{trial}_mixing_and_scaler.npz", matrices=np.stack(matrices),
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        initial_hashes = []
        for arm in ARMS:
            print(f"任务五 第1步：trial={trial}, shifted_mean={mu:.4f}, arm={arm}", flush=True)
            model, history, initial_hash = fit_model(train_x, arm, epochs, batch_size, SEED+trial)
            initial_hashes.append(initial_hash)
            info = {"trial": trial, "shifted_mean": float(mu), "arm": arm}
            rows, diagnostics = diagnose(model, train_x, test_x, train_z, test_z, info)
            all_rows.extend(rows)
            all_diagnostics.extend(diagnostics)
            all_history.extend([{**info, **record} for record in history])
            torch.save({"state_dict": model.state_dict(), "info": info,
                        "metadata_file": "experiment_metadata.json"}, args.output / f"trial_{trial}_{arm}.pt")
        if initial_hashes[0] != initial_hashes[1]:
            raise RuntimeError("两组初始权重不一致。")
        checks.append({"trial": trial, "same_initial_weights": True,
                       "training_receives_latent_truth": False, "passed": True})
        for name, rows in (("latent_recovery.csv", all_rows), ("representation_diagnostics.csv", all_diagnostics),
                           ("training_history.csv", all_history), ("control_checks.csv", checks)):
            pd.DataFrame(rows).to_csv(args.output / name, index=False, encoding="utf-8-sig")
    if len(all_rows) != 72 or len(all_diagnostics) != 12 or len(checks) != 3:
        raise RuntimeError("试验数量不完整，停止生成完成标记。")
    if not np.isfinite(pd.DataFrame(all_rows).test_r2.to_numpy()).all():
        raise RuntimeError("读出器评估存在非有限值。")
    draw_figures(pd.DataFrame(all_rows), pd.DataFrame(all_diagnostics), pd.DataFrame(all_history), args.output)
    (args.output / "completion.json").write_text(json.dumps({"completed": True, "training_runs": 6,
                                                            "reproduction_level": metadata["reproduction_level"]}), encoding="utf-8")
    print(f"任务五 第1步完成，请提交整个输出目录：{args.output}")


if __name__ == "__main__":
    main()
