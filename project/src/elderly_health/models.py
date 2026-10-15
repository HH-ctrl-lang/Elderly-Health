"""Shared V4/V5 encoders and prediction heads; no dataset or device globals."""

import torch
from torch import nn

LATENT_DIM = 32
GATE_TEMPERATURE = 5.0


class CausalGate(nn.Module):
    def __init__(self, input_dim, init_std=0.1, temperature=5.0):
        super().__init__()
        self.raw_mask = nn.Parameter(torch.randn(input_dim) * init_std)
        self.temperature = temperature

    def forward(self):
        return torch.sigmoid(self.temperature * self.raw_mask)

    def get_scores(self):
        return torch.sigmoid(self.temperature * self.raw_mask).detach()


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z)


class EnvClassifier(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_envs=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_envs),
        )

    def forward(self, z):
        return self.net(z)


class CausalDisentangledModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_envs,
        latent_dim=LATENT_DIM,
        gate_init_std=0.1,
        gate_temperature=5.0,
        joint=True,
    ):
        super().__init__()
        self.joint = joint
        self.gate = CausalGate(input_dim, init_std=gate_init_std, temperature=gate_temperature)
        self.encoder_c = Encoder(input_dim, latent_dim)
        self.encoder_s = Encoder(input_dim, latent_dim)
        self.predictor_zc = Predictor(latent_dim)
        self.predictor_joint = Predictor(latent_dim * 2)
        self.env_cls = EnvClassifier(latent_dim, num_envs)

    def forward(self, x):
        mask = self.gate()
        x_c = x * mask
        x_s = x * (1.0 - mask)
        z_c = self.encoder_c(x_c)
        z_s = self.encoder_s(x_s)

        joint_z = torch.cat([z_c, z_s], dim=1)
        logits = self.predictor_joint(joint_z) if self.joint else self.predictor_zc(z_c)
        env_logits = self.env_cls(z_s)
        return logits, env_logits, z_c, z_s, mask

    def predict(self, x):
        mask = self.gate()

        z_c = self.encoder_c(x * mask)
        z_s = self.encoder_s(x * (1.0 - mask))
        joint_z = torch.cat([z_c, z_s], dim=1)

        return self.predictor_joint(joint_z) if self.joint else self.predictor_zc(z_c)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
