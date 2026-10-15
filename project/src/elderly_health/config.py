"""Validated configuration shared by training and cross-environment evaluation."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    model: str = "v5"
    seed: int = 42
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 80
    latent_dim: int = 32
    pos_weight: float = 3.0
    sparse_start: int = 20
    irm_start: int = 50
    ramp_epochs: int = 40
    irm_grace_epochs: int = 50
    irm_mode: str = "v11"
    independence: str = "hsic"
    sparsity: str = "target"
    target_sparsity: float = 5.0 / 19.0
    lambda_sparse: float = 0.3
    lambda_entropy: float = 30.0
    lambda_irm: float = 0.01
    lambda_env: float = 1.0
    lambda_independence: float = 0.02
    lambda_graph: float = 0.1
    gate_init_std: float = 0.02
    gate_temperature: float = 5.0
    hsic_sigma: float = 1.0
    device: str = "cpu"
    threads: int = 2
    probe_epochs: int = 30

    def __post_init__(self):
        for name in (
            "seed",
            "epochs",
            "batch_size",
            "patience",
            "latent_dim",
            "threads",
            "sparse_start",
            "irm_start",
            "ramp_epochs",
            "irm_grace_epochs",
            "probe_epochs",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        choices = {
            "model": {"baseline", "v4", "v5"},
            "irm_mode": {"none", "standard", "v11"},
            "independence": {"none", "cosine", "hsic"},
            "sparsity": {"none", "target", "mean"},
            "device": {"cpu", "cuda", "auto"},
        }
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError(f"{name} must be one of {sorted(values)}")
        for name in ("epochs", "batch_size", "patience", "latent_dim", "threads", "ramp_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.sparse_start <= self.irm_start:
            raise ValueError("Require 0 <= sparse_start <= irm_start")
        if not 0 <= self.target_sparsity <= 1:
            raise ValueError("target_sparsity must be between zero and one")
        for name in ("learning_rate", "pos_weight", "gate_temperature", "hsic_sigma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "weight_decay",
            "lambda_sparse",
            "lambda_entropy",
            "lambda_irm",
            "lambda_env",
            "lambda_independence",
            "lambda_graph",
            "gate_init_std",
            "irm_grace_epochs",
            "probe_epochs",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self):
        return asdict(self)
