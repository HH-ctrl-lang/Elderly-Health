"""CPU-friendly tabular VREx adapter, based on DomainBed's VREx objective.

Source: facebookresearch/DomainBed commit b93c22a1cfc3b2428398272c1a116c8de1f4139e,
domainbed/algorithms.py, class VREx; MIT license.
This is a *tabular adaptation*, not a reproduction of DomainBed's original
image training pipeline or its default hyperparameters.
"""

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .models import SimpleMLP
from .training import get_device, seed_everything


DOMAINBED_SHA = "b93c22a1cfc3b2428398272c1a116c8de1f4139e"


@dataclass(frozen=True)
class VRExSettings:
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    epochs: int = 300
    per_environment_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    positive_weight: float = 3.0
    vrex_lambda: float = 10.0
    penalty_anneal_steps: int = 500
    patience: int = 80
    selection_start_epoch: int = 91

    def __post_init__(self):
        if self.epochs < 1 or self.selection_start_epoch < 1:
            raise ValueError("Epochs and selection start must be positive")
        if self.selection_start_epoch > self.epochs:
            raise ValueError("Selection start cannot exceed epochs")
        if self.per_environment_batch_size < 1 or self.threads < 1 or self.patience < 1:
            raise ValueError("Batch size, threads and patience must be positive")
        if self.penalty_anneal_steps < 0:
            raise ValueError("Penalty anneal steps must be nonnegative")
        for value in (self.learning_rate, self.positive_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and positive weight must be finite and positive")
        for value in (self.weight_decay, self.vrex_lambda):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Regularization parameters must be finite and nonnegative")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("Unsupported device")

    def to_dict(self):
        return asdict(self)


def vrex_risk_loss(logits, targets, environment_ids, positive_weight=3.0, penalty_weight=10.0):
    """Mean domain risk plus variance of domain risks, following DomainBed VREx.

    DomainBed uses multi-class cross-entropy. Here BCEWithLogitsLoss with an
    explicit positive-class weight handles the project's binary labels and
    preserves its existing SimpleMLP prediction head.
    """
    if logits.shape != targets.shape:
        raise ValueError("Expected logits and binary targets with identical shapes")
    per_domain = []
    pos_weight = logits.new_tensor([positive_weight])
    for env in torch.unique(environment_ids):
        mask = environment_ids == env
        per_domain.append(
            F.binary_cross_entropy_with_logits(
                logits[mask], targets[mask], pos_weight=pos_weight
            )
        )
    if len(per_domain) < 2:
        raise ValueError("VREx requires at least two nonempty training domains")
    risks = torch.stack(per_domain)
    mean_risk = risks.mean()
    variance = (risks - mean_risk).square().mean()
    return mean_risk + penalty_weight * variance, mean_risk, variance


def fit_vrex(train, val, settings):
    """Fit VREx with environment-balanced minibatches and validation-only selection."""
    seed_everything(settings.seed, settings.threads)
    device = get_device(settings.device)
    domains = sorted(set(map(str, train.env)))
    if len(domains) < 2:
        raise ValueError("VREx requires at least two source environments")
    if len(set(train.y)) < 2 or len(set(val.y)) < 2:
        raise ValueError("Train and validation must contain both outcome classes")

    model = SimpleMLP(train.x.shape[1]).to(device)
    domain_positions = [
        np.flatnonzero(np.asarray(train.env, dtype=str) == domain) for domain in domains
    ]
    rng = torch.Generator().manual_seed(settings.seed)
    steps_per_epoch = max(
        1,
        math.ceil(len(train.x) / (len(domains) * settings.per_environment_batch_size)),
    )

    def new_optimizer():
        return torch.optim.Adam(
            model.parameters(), lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    optimizer = new_optimizer()
    val_x = torch.as_tensor(val.x, dtype=torch.float32, device=device)
    step = 0
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    history = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_risk = total_penalty = 0.0
        for _ in range(steps_per_epoch):
            if step == settings.penalty_anneal_steps:
                # DomainBed resets Adam at the regularization weight transition.
                optimizer = new_optimizer()
            batch_ids = []
            batch_envs = []
            for env_id, positions in enumerate(domain_positions):
                draw = torch.randint(
                    0, len(positions), (settings.per_environment_batch_size,),
                    generator=rng,
                ).numpy()
                chosen = positions[draw]
                batch_ids.append(chosen)
                batch_envs.append(np.full(len(chosen), env_id, dtype=np.int64))
            indices = np.concatenate(batch_ids)
            x = torch.as_tensor(train.x[indices], dtype=torch.float32, device=device)
            y = torch.as_tensor(train.y[indices, None], dtype=torch.float32, device=device)
            env = torch.as_tensor(np.concatenate(batch_envs), device=device)
            active_weight = (
                settings.vrex_lambda if step >= settings.penalty_anneal_steps else 1.0
            )
            objective, mean_risk, penalty = vrex_risk_loss(
                model(x), y, env, settings.positive_weight, active_weight
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite VREx objective at step {step}")
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            total_risk += mean_risk.detach().item()
            total_penalty += penalty.detach().item()
            step += 1

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_x)).cpu().numpy().ravel()
        val_auc = float(roc_auc_score(val.y, val_prob))
        selectable = epoch >= settings.selection_start_epoch
        history.append({
            "epoch": epoch, "updates": step, "val_auc": val_auc,
            "mean_risk": total_risk / steps_per_epoch,
            "risk_variance": total_penalty / steps_per_epoch,
            "eligible_for_selection": selectable,
        })
        if selectable:
            if val_auc > best_auc:
                best_auc, best_epoch, stale = val_auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        if epoch % 30 == 0 or epoch == 1:
            print(f"VREx epoch {epoch}: val_auc={val_auc:.4f}", flush=True)
        if selectable and stale >= settings.patience:
            break

    if best_state is None:
        raise RuntimeError("No eligible VREx checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "history": history,
        "training_domains": domains,
        "steps_per_epoch": steps_per_epoch,
        "domainbed_source_sha": DOMAINBED_SHA,
    }
