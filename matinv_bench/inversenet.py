"""InverseNet: neural models that map a matrix to its inverse.

Two architectures are provided:

* ``InverseNetMLP`` — the classic "InverseNet" formulation: flatten the
  matrix and regress the flattened inverse with a fully connected network.
  This is only practical for small dimensions; at n=500 the input/output
  layers alone would need >2B parameters, so the MLP is trained for
  n=10 and n=100 only.

* ``InverseNetNS`` — a learned Newton–Schulz iteration. The classic
  iteration ``X_{k+1} = X_k (2I - A X_k)`` is unrolled for a fixed number
  of steps and the scalar coefficients of every step (plus the scale of
  the initial guess ``X_0 = alpha * A^T / (||A||_1 ||A||_inf)``) are
  learned. The parameter count is independent of the matrix dimension,
  so it scales to n=500 and beyond.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InverseNetMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        d = dim * dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        b = a.shape[0]
        out = self.net(a.reshape(b, -1))
        return out.reshape(b, self.dim, self.dim)


class InverseNetNS(nn.Module):
    def __init__(self, num_steps: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.full((num_steps,), 2.0))
        self.gamma = nn.Parameter(torch.ones(num_steps))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # ||A||_1 * ||A||_inf upper-bounds ||A||_2^2, giving a contractive X_0.
        norm1 = a.abs().sum(dim=-2).max(dim=-1).values
        norminf = a.abs().sum(dim=-1).max(dim=-1).values
        scale = (self.alpha / (norm1 * norminf)).reshape(-1, 1, 1)
        x = scale * a.transpose(-2, -1)
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        for k in range(self.num_steps):
            x = x @ (self.beta[k] * eye - self.gamma[k] * (a @ x))
        return x


def build_model(kind: str, dim: int) -> nn.Module:
    if kind == "mlp":
        hidden = {10: 512, 100: 1024}.get(dim)
        if hidden is None:
            raise ValueError(f"InverseNetMLP is not practical for dim={dim}")
        return InverseNetMLP(dim, hidden)
    if kind == "ns":
        return InverseNetNS(num_steps=8)
    raise ValueError(f"unknown model kind: {kind}")
