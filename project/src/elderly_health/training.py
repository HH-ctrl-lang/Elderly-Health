"""Deterministic training, source-only validation, and in-memory best checkpoints."""

import copy
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FEATURES
from .graph import graph_guided_loss
from .losses import independence_loss, irm_penalty
from .models import CausalDisentangledModel, SimpleMLP


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
    )


def ramp(epoch, start, span):
    return max(0.0, min(1.0, (epoch - start) / span))


def prediction_logits(model, x, head="main"):
    if isinstance(model, SimpleMLP):
        return model(x)
    if head == "zc":
        return model.predictor_zc(model.encoder_c(x * model.gate()))
    return model.predict(x)


def predict(model, x, device, batch_size=256, head="main"):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], device=device)
            probabilities.append(torch.sigmoid(prediction_logits(model, batch, head)).cpu().numpy())
    return np.concatenate(probabilities).ravel()


def fit(train, val, config, prior=None):
    seed_everything(config.seed, config.threads)
    device = get_device(config.device)
    env_map = {str(name): i for i, name in enumerate(sorted(set(train.env)))}
    if set(val.env) - set(env_map):
        raise ValueError("Validation contains environments absent from training")
    if len(set(train.y)) != 2 or len(set(val.y)) != 2:
        raise ValueError("Training and validation both require positive and negative samples")

    if config.model == "baseline":
        model = SimpleMLP(train.x.shape[1]).to(device)
    else:
        model = CausalDisentangledModel(
            train.x.shape[1],
            len(env_map),
            config.latent_dim,
            config.gate_init_std,
            gate_temperature=config.gate_temperature,
            joint=config.model == "v5",
        ).to(device)
    dataset = TensorDataset(
        torch.as_tensor(train.x),
        torch.as_tensor(train.y).unsqueeze(1),
        torch.tensor([env_map[e] for e in train.env], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight], device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )
    val_x = torch.as_tensor(val.x, device=device)
    val_y = torch.as_tensor(val.y, device=device).unsqueeze(1)
    best_auc, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    history = []
    causal = config.model != "baseline"

    for epoch in range(config.epochs):
        # Select the best model after all scheduled objectives enter, if that phase is reached.
        if causal and epoch == config.irm_start:
            best_auc, best_state, stale = -float("inf"), None, 0
        model.train()
        totals = {}
        for x, y, environment in loader:
            x, y, environment = x.to(device), y.to(device), environment.to(device)
            if not causal:
                components = {"prediction": criterion(model(x), y)}
                loss = components["prediction"]
            else:
                logits, env_logits, zc, zs, mask = model(x)
                components = {"prediction": criterion(logits, y)}
                loss = components["prediction"]
                if config.model == "v5":
                    components["zc_prediction"] = criterion(model.predictor_zc(zc), y)
                    loss = loss + components["zc_prediction"]
                sparse_ramp = ramp(epoch, config.sparse_start, config.ramp_epochs)
                invariant_ramp = ramp(epoch, config.irm_start, config.ramp_epochs)
                if sparse_ramp > 0:
                    if config.sparsity == "target":
                        sparse = (mask.mean() - config.target_sparsity).abs()
                    elif config.sparsity == "mean":
                        sparse = mask.mean()
                    else:
                        sparse = mask.sum() * 0.0
                    components["sparse"] = sparse
                    components["entropy"] = (mask * (1 - mask)).mean()
                    loss = loss + sparse_ramp * (
                        config.lambda_sparse * sparse
                        + config.lambda_entropy * components["entropy"]
                    )
                    if prior is not None and config.lambda_graph > 0:
                        components["graph"] = graph_guided_loss(mask, FEATURES, prior)
                        loss = loss + sparse_ramp * config.lambda_graph * components["graph"]
                if invariant_ramp > 0:
                    components["irm"] = irm_penalty(model, zc, y, environment, config.irm_mode)
                    components["environment"] = nn.functional.cross_entropy(env_logits, environment)
                    components["independence"] = independence_loss(
                        zc,
                        zs,
                        config.independence,
                        sigma=config.hsic_sigma,
                    )
                    loss = loss + invariant_ramp * (
                        config.lambda_irm * components["irm"]
                        + config.lambda_env * components["environment"]
                        + config.lambda_independence * components["independence"]
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            components["total"] = loss
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * len(x)

        model.eval()
        with torch.no_grad():
            logits = prediction_logits(model, val_x)
            val_loss = criterion(logits, val_y).item()
            auc = roc_auc_score(val.y, torch.sigmoid(logits).cpu().numpy().ravel())
        scheduler.step(val_loss)
        record = {"epoch": epoch + 1, "val_auc": float(auc), "val_loss": val_loss}
        record.update({f"train_{k}": v / len(train.x) for k, v in totals.items()})
        if causal:
            record["mask"] = model.gate.get_scores().cpu().tolist()
        history.append(record)
        if auc > best_auc:
            best_auc, best_epoch, stale = float(auc), epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 30 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch + 1}: validation AUC={auc:.4f}; loss={val_loss:.4f}", flush=True)
        earliest_stop = config.irm_start + config.irm_grace_epochs if causal else 0
        if epoch >= earliest_stop and stale >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "history": history,
        "env_map": env_map,
        "device": str(device),
    }
