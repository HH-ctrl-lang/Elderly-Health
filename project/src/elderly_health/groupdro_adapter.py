"""GroupDRO tabular CPU adaptation of Sagawa et al.'s official loss update.

Official code: https://github.com/kohpangwei/group_DRO
Commit cbbc1c5b06844e46b87e264326b56056d2a437d1, loss.py,
LossComputer.compute_robust_loss. MIT license in GROUPDRO_LICENSE.txt.

Preserved: multiplicative adversarial-group-probability update and robust risk
as the weighted sum of the source-group risks. Necessary project adaptations:
19-input SimpleMLP, weighted binary BCE, CPU-safe grouping, existing source-only
imputation/splits, and validation-based checkpoint selection. The original
CelebA/Waterbirds/MultiNLI benchmark is not reproduced.
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


GROUPDRO_SOURCE_SHA = "cbbc1c5b06844e46b87e264326b56056d2a437d1"


@dataclass(frozen=True)
class GroupDROSettings:
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    epochs: int = 300
    per_environment_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    positive_weight: float = 3.0
    robust_step_size: float = 0.01
    patience: int = 80
    selection_start_epoch: int = 91

    def __post_init__(self):
        for name in ("threads", "epochs", "per_environment_batch_size", "patience", "selection_start_epoch"):
            x = getattr(self, name)
            if type(x) is not int or x < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.selection_start_epoch > self.epochs:
            raise ValueError("selection_start_epoch cannot exceed epochs")
        for name in ("learning_rate", "positive_weight", "robust_step_size"):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.device not in ("cpu", "auto", "cuda"):
            raise ValueError("Unknown device")

    def to_dict(self):
        return asdict(self)


class GroupDRO(nn.Module):
    """Original GroupDRO robust-risk update with a project binary head."""

    def __init__(self, input_dim, num_groups, settings):
        super().__init__()
        if num_groups < 2:
            raise ValueError("GroupDRO needs at least two groups")
        self.network = SimpleMLP(input_dim)
        self.hparams = settings
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        # The neural architecture/optimizer are tabular-data adaptations.
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        return self.network(x)

    def update(self, minibatches):
        """Each pair (x, y) contains data from exactly one source group."""
        if len(minibatches) != len(self.adv_probs):
            raise ValueError("Expected one nonempty minibatch for every training group")
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        pos_weight = all_logits.new_tensor([self.hparams.positive_weight])
        risks = []
        start = 0
        for x, y in minibatches:
            if len(x) < 1:
                raise ValueError("Empty training group minibatch")
            logits = all_logits[start : start + len(x)]
            start += len(x)
            risks.append(
                F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight
                )
            )
        group_losses = torch.stack(risks)

        # Official loss.py: adv_probs *= exp(eta * group_loss.data), normalize.
        # Subtracting a common maximum before exp is algebraically equivalent
        # but prevents overflow when adapting the authors' code to long runs.
        with torch.no_grad():
            adjusted = self.hparams.robust_step_size * group_losses.detach()
            weights = self.adv_probs * torch.exp(adjusted - adjusted.max())
            self.adv_probs.copy_(weights / weights.sum())

        robust_loss = torch.dot(group_losses, self.adv_probs)
        if not torch.isfinite(robust_loss):
            raise FloatingPointError("Nonfinite GroupDRO objective")

        self.optimizer.zero_grad()
        robust_loss.backward()
        self.optimizer.step()
        return {
            "robust_loss": float(robust_loss.detach()),
            "mean_group_loss": float(group_losses.detach().mean()),
            "worst_group_loss": float(group_losses.detach().max()),
            "group_losses": group_losses.detach().cpu().tolist(),
            "group_weights": self.adv_probs.detach().cpu().tolist(),
        }


def fit_groupdro(train, val, settings):
    seed_everything(settings.seed, settings.threads)
    device = get_device(settings.device)
    domains = sorted(set(map(str, train.env)))
    if len(domains) < 2:
        raise ValueError("GroupDRO needs two or more source groups")
    if len(set(train.y)) != 2 or len(set(val.y)) != 2:
        raise ValueError("Training and validation must contain both outcome classes")

    model = GroupDRO(train.x.shape[1], len(domains), settings).to(device)
    positions = [
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
        total_robust = total_mean = total_worst = 0.0
        for _ in range(steps_per_epoch):
            minibatches = []
            for group_rows in positions:
                indices = group_rows[
                    torch.randint(
                        0,
                        len(group_rows),
                        (settings.per_environment_batch_size,),
                        generator=generator,
                    ).numpy()
                ]
                x = torch.as_tensor(train.x[indices], dtype=torch.float32, device=device)
                y = torch.as_tensor(train.y[indices, None], dtype=torch.float32, device=device)
                minibatches.append((x, y))
            result = model.update(minibatches)
            total_robust += result["robust_loss"]
            total_mean += result["mean_group_loss"]
            total_worst += result["worst_group_loss"]

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_x)).cpu().numpy().ravel()
        auc = float(roc_auc_score(val.y, val_prob))
        selectable = epoch >= settings.selection_start_epoch
        history.append({
            "epoch": epoch,
            "updates": epoch * steps_per_epoch,
            "val_auc": auc,
            "eligible_for_selection": selectable,
            "robust_loss": total_robust / steps_per_epoch,
            "mean_group_loss": total_mean / steps_per_epoch,
            "worst_group_loss": total_worst / steps_per_epoch,
            "group_weights": model.adv_probs.detach().cpu().tolist(),
        })
        if selectable:
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        if epoch == 1 or epoch % 30 == 0:
            print(f"GroupDRO epoch {epoch}: validation AUC={auc:.4f}", flush=True)
        if selectable and stale >= settings.patience:
            break

    if best_state is None:
        raise RuntimeError("No eligible GroupDRO checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "training_groups": domains,
        "steps_per_epoch": steps_per_epoch,
        "source_commit": GROUPDRO_SOURCE_SHA,
        "selected_group_weights": model.adv_probs.detach().cpu().tolist(),
        "history": history,
    }
