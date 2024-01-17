from typing import Dict

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, tau: float):
        super(HuberLoss, self).__init__()
        self.register_buffer("tau", torch.tensor(tau), persistent=False)

    def forward(self, x):
        x = x.abs()
        return torch.where(x.le(self.tau),
                           x ** 2 / 2,
                           self.tau ** 2 / 2 + self.tau * (x - self.tau))


class PowerSeries(nn.Module):
    def __init__(self, value: float, power_min, power_max, tau: float = 1.):
        super(PowerSeries, self).__init__()
        self.value = value

        # compute weights vector
        powers = torch.arange(power_min, power_max)
        self.register_buffer("weights", self.value ** powers, persistent=False)

        self.dim = len(self.weights)
        self.loss_fn = HuberLoss(tau)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor,
                nlog_c1: torch.Tensor | None = None, nlog_c2: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        r"""x2[i] is the pitch-shifted version of x1[i] by target[i] semitones, i.e.
            if x1[i] is a C# and x2[i] is a C then target[i] = -1

        """
        z1 = self.project(x1)
        z2 = self.project(x2)
        if nlog_c1 is not None:
            z1 = z1 * torch.exp(-nlog_c1)
        if nlog_c2 is not None:
            z2 = z2 * torch.exp(-nlog_c2)

        # compute frequency ratios out of semitones
        freq_ratios = self.value ** target.float()

        # compute equivariant loss
        loss_12 = self.loss_fn(z2 / z1 - freq_ratios).mean()
        loss_21 = self.loss_fn(z1 / z2 - 1/freq_ratios).mean()

        return (loss_12 + loss_21) / 2

    def project(self, x: torch.Tensor):
        r"""Projects a batch of vectors into a batch of scalars
        Args:
            x (torch.Tensor): batch of input vectors, shape (batch_size, output_dim)

        Returns:
            torch.Tensor: batch of output scalars, shape (batch_size)
        """
        return x.mv(self.weights)
