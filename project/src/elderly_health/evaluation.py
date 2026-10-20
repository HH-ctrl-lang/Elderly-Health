"""Separate prediction, graph agreement, and representation probe diagnostics."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import FEATURES
from .graph import prior_jaccard
from .models import SimpleMLP
from .training import get_device, predict


def metrics(y, probabilities, threshold=0.5):
    labels = probabilities > threshold
    both_classes = len(np.unique(y)) == 2
    return {
        "auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "auprc": float(average_precision_score(y, probabilities)) if y.sum() else None,
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)) if both_classes else None,
        "f1": float(f1_score(y, labels, zero_division=0)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "n": len(y),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
    }


def representations(model, split, device):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(split.x, device=device)
        mask = model.gate()
        return (
            model.encoder_c(x * mask).cpu().numpy(),
            model.encoder_s(x * (1 - mask)).cpu().numpy(),
        )


def environment_probe(z_train, env_train, z_test, env_test, config):
    """Fit on source training representations; unseen test labels are excluded explicitly."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    mapping = {e: i for i, e in enumerate(sorted(set(env_train)))}
    known = np.array([e in mapping for e in env_test])
    result = {"n_known": int(known.sum()), "n_unseen": int((~known).sum())}
    if not known.any():
        return {**result, "accuracy": None, "balanced_accuracy": None, "majority_accuracy": None}
    # Probe randomness must not depend on which model/metric was evaluated first.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 1000)
        probe = nn.Sequential(
            nn.Linear(z_train.shape[1], 32), nn.ReLU(), nn.Linear(32, len(mapping))
        )
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
        dataset = TensorDataset(
            torch.tensor(z_train), torch.tensor([mapping[e] for e in env_train])
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + 1000),
        )
        for _ in range(config.probe_epochs):
            for z, e in loader:
                optimizer.zero_grad()
                nn.functional.cross_entropy(probe(z), e).backward()
                optimizer.step()
        with torch.no_grad():
            predictions = probe(torch.tensor(z_test[known])).argmax(1).numpy()
    truth = np.array([mapping[e] for e in env_test[known]])
    majority = max(mapping, key=lambda e: int((env_train == e).sum()))
    return {
        **result,
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "majority_accuracy": float((env_test[known] == majority).mean()),
    }


def evaluate(model, splits, config, prior=None):
    device = get_device(config.device)
    test = splits[2]
    probabilities = predict(model, test.x, device)
    result = {"main": metrics(test.y, probabilities)}
    result["per_environment"] = {
        str(env): metrics(test.y[test.env == env], probabilities[test.env == env])
        for env in sorted(set(test.env))
    }
    if test.site_env is not None and not np.array_equal(test.env, test.site_env):
        result["per_site"] = {
            str(site): metrics(
                test.y[test.site_env == site], probabilities[test.site_env == site]
            )
            for site in sorted(set(test.site_env))
        }
    if not isinstance(model, SimpleMLP):
        result["zc"] = metrics(test.y, predict(model, test.x, device, head="zc"))
        mask = model.gate.get_scores().cpu().numpy()
        result["gate"] = {name: float(value) for name, value in zip(FEATURES, mask)}
        result["selected_features"] = [name for name, value in zip(FEATURES, mask) if value > 0.5]
        result["graph_jaccard"] = prior_jaccard(mask, FEATURES, prior)
        if config.probe_epochs:
            zc_train, zs_train = representations(model, splits[0], device)
            zc_test, zs_test = representations(model, test, device)
            result["zc_env_probe"] = environment_probe(
                zc_train,
                splits[0].env,
                zc_test,
                test.env,
                config,
            )
            result["zs_env_probe"] = environment_probe(
                zs_train,
                splits[0].env,
                zs_test,
                test.env,
                config,
            )
    return result


def export_predictions(model, splits, config, output):
    device = get_device(config.device)
    for name, split in zip(("train", "val", "test"), splits):
        table = pd.DataFrame(
            {
                "sample_id": split.sample_ids,
                "environment_code": (
                    split.site_env if split.site_env is not None else split.env
                ),
                "environment_group": split.env,
                "cognitive_impairment": split.y.astype(int),
                "split": name,
                "seed": config.seed,
                "y_prob": predict(model, split.x, device),
            }
        )
        if not isinstance(model, SimpleMLP):
            zc, zs = representations(model, split, device)
            table["zc_prob"] = predict(model, split.x, device, head="zc")
            for prefix, values in (("zc", zc), ("zs", zs)):
                for column in range(values.shape[1]):
                    table[f"{prefix}_{column}"] = values[:, column]
        table.to_csv(output / f"representations_{name}.csv", index=False)


def save_curves(y, probabilities, output):
    """Save ROC and precision-recall curves without opening a GUI window."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    if len(np.unique(y)) != 2:
        return
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    for name, x, vertical, xlabel, ylabel in (
        ("roc", fpr, tpr, "False positive rate", "True positive rate"),
        ("precision_recall", recall, precision, "Recall", "Precision"),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(x, vertical)
        if name == "roc":
            axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
        else:
            axis.axhline(float(np.mean(y)), color="gray", linestyle="--")
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)
