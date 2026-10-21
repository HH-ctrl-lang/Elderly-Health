"""RDM domain-generalization objective adapted from the authors' original code.

Original: Nguyen et al., "Domain Generalisation via Risk Distribution Matching",
WACV 2024, https://github.com/nktoan/risk-distribution-matching
At commit 1b8830e1b373ccacc7b50a2089ebfc2c188bb6b2,
DomainBed/domainbed/algorithms.py, class RDM. MIT license (see RDM_LICENSE.txt).

Preserved from original: worst mean-risk environment, multiscale Gaussian-kernel
MMD comparing its individual sample losses with pooled sample losses, optional
variance penalty, delayed penalty activation, and Adam reset at activation.

Changes required for the elderly-health data: use the project's 19-input
SimpleMLP, binary BCEWithLogits per-sample loss, CPU/device-safe tensors,
source-only validation and existing site/region data splits. This is a
tabular adaptation, not a reproduction of the image benchmark results.
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


RDM_SOURCE_SHA = "1b8830e1b373ccacc7b50a2089ebfc2c188bb6b2"


@dataclass(frozen=True)
class RDMSettings:
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    epochs: int = 300
    per_environment_batch_size: int = 32
    learning_rate: float = 3e-4
    rdm_lr: float = 1.5e-5
    weight_decay: float = 1e-5
    positive_weight: float = 3.0
    rdm_lambda: float = 5.0
    rdm_penalty_anneal_iters: int = 1500
    variance_weight: float = 0.0
    patience: int = 80
    selection_start_epoch: int = 91

    def __post_init__(self):
        for name in ("epochs", "threads", "per_environment_batch_size", "patience", "selection_start_epoch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("rdm_penalty_anneal_iters", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.selection_start_epoch > self.epochs:
            raise ValueError("selection_start_epoch must not exceed epochs")
        for name in ("learning_rate", "rdm_lr", "positive_weight"):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "rdm_lambda", "variance_weight"):
            val = getattr(self, name)
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")

    def to_dict(self):
        return asdict(self)


class RDM(nn.Module):
    """Author's RDM loss/update adapted only for tabular binary classification."""

    def __init__(self, input_dim, settings):
        super().__init__()
        self.network = SimpleMLP(input_dim)
        self.hparams = settings
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))
        self.optimizer = self._adam(settings.learning_rate)

    def _adam(self, lr):
        return torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=self.hparams.weight_decay
        )

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.network(x)

    # The next three methods follow the original RDM class's Gaussian MMD.
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(
        self, x, y, gamma=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
    ):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches):
        """Each minibatch is a (X, binary label) pair for one source environment."""
        if len(minibatches) < 2:
            raise ValueError("RDM needs at least two source environments")
        step = int(self.update_count.item())
        matching_penalty_weight = (
            self.hparams.rdm_lambda
            if step >= self.hparams.rdm_penalty_anneal_iters else 0.0
        )
        variance_penalty_weight = (
            self.hparams.variance_weight
            if step >= self.hparams.rdm_penalty_anneal_iters else 0.0
        )
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        all_logits_idx = 0
        losses, all_confs_envs = [], []
        pos_weight = all_logits.new_tensor([self.hparams.positive_weight])
        for x, y in minibatches:
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            sample_losses = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none"
            ).reshape(-1)
            losses.append(sample_losses.mean())
            all_confs_envs.append(sample_losses)

        # Pooling all individual sample risks; group sizes are equal by design.
        risks = torch.stack(losses)
        erm_loss = risks.mean()
        worst_env_idx = int(torch.argmax(risks.detach()).item())
        all_confs_worst_env_flat = all_confs_envs[worst_env_idx]
        all_confs_all_envs_flat = torch.cat(all_confs_envs)

        matching_penalty = self.mmd(
            all_confs_worst_env_flat.unsqueeze(1),
            all_confs_all_envs_flat.unsqueeze(1),
        )
        variance_penalty = (
            torch.var(all_confs_all_envs_flat)
            + torch.var(all_confs_worst_env_flat)
        )
        total_loss = (
            erm_loss
            + matching_penalty_weight * matching_penalty
            + variance_penalty_weight * variance_penalty
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Nonfinite RDM objective at update {step}")

        if step == self.hparams.rdm_penalty_anneal_iters:
            # Same objective-transition optimizer reset as authors' code.
            self.optimizer = self._adam(self.hparams.rdm_lr)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_count.add_(1)
        return {
            "total_loss": float(total_loss.detach().item()),
            "erm_loss": float(erm_loss.detach().item()),
            "matching_penalty": float(matching_penalty.detach().item()),
            "variance_penalty": float(variance_penalty.detach().item()),
            "matching_weight": matching_penalty_weight,
        }


def fit_rdm(train, val, settings):
    """Train with domain-balanced minibatches and validation-only checkpoints."""
    seed_everything(settings.seed, settings.threads)
    device = get_device(settings.device)
    domains = sorted(set(map(str, train.env)))
    if len(domains) < 2:
        raise ValueError("RDM needs two or more training environments")
    if len(set(train.y)) != 2 or len(set(val.y)) != 2:
        raise ValueError("Training and validation both need two outcome classes")

    model = RDM(train.x.shape[1], settings).to(device)
    domain_positions = [
        np.flatnonzero(np.asarray(train.env, dtype=str) == domain)
        for domain in domains
    ]
    generator = torch.Generator().manual_seed(settings.seed)
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train.x) / (len(domains) * settings.per_environment_batch_size)
        ),
    )
    val_x = torch.as_tensor(val.x, dtype=torch.float32, device=device)
    best_auc, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    history = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        totals = {}
        for _ in range(steps_per_epoch):
            minibatches = []
            for positions in domain_positions:
                chosen = positions[
                    torch.randint(
                        0, len(positions),
                        (settings.per_environment_batch_size,),
                        generator=generator,
                    ).numpy()
                ]
                x = torch.as_tensor(
                    train.x[chosen], dtype=torch.float32, device=device
                )
                y = torch.as_tensor(
                    train.y[chosen, None], dtype=torch.float32, device=device
                )
                minibatches.append((x, y))
            record = model.update(minibatches)
            for key, value in record.items():
                totals[key] = totals.get(key, 0.0) + value
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_x)).cpu().numpy().ravel()
        auc = float(roc_auc_score(val.y, probs))
        selectable = epoch >= settings.selection_start_epoch
        history.append({
            "epoch": epoch,
            "updates": int(model.update_count.item()),
            "val_auc": auc,
            "eligible_for_selection": selectable,
            **{key: value / steps_per_epoch for key, value in totals.items()},
        })
        if selectable:
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        if epoch == 1 or epoch % 30 == 0:
            print(f"RDM epoch {epoch}: validation AUC={auc:.4f}", flush=True)
        if selectable and stale >= settings.patience:
            break
    if best_state is None:
        raise RuntimeError("No eligible RDM checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "training_domains": domains,
        "steps_per_epoch": steps_per_epoch,
        "source_commit": RDM_SOURCE_SHA,
        "history": history,
    }
