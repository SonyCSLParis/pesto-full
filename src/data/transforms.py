from typing import Optional

import torch
import torch.nn as nn


class ToLogMagnitude(nn.Module):
    def __init__(self):
        super(ToLogMagnitude, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        if x.size(-1) == 2:
            x = torch.view_as_complex(x)
        if x.ndim == 2:
            x.unsqueeze_(1)
        x = x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)
        return x


class BatchRandomNoise(nn.Module):
    def __init__(
            self,
            min_snr: float = 0.0001,
            max_snr: float = 0.01,
            p: Optional[float] = None,
    ):
        super(BatchRandomNoise, self).__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        snr = torch.empty(batch_size, device=device)
        snr.uniform_(self.min_snr, self.max_snr)
        mask = torch.rand_like(snr).le(self.p)
        snr[mask] = 0

        noise_std = snr * x.view(batch_size, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(batch_size, -1)).view_as(x)

        # compute ratios corresponding to gain in dB
        noise = noise_std * torch.randn_like(x)

        return x + noise


class BatchRandomGain(nn.Module):
    def __init__(
            self,
            min_gain: float = 0.5,
            max_gain: float = 1.5,
            p: Optional[float] = None,
    ):
        super(BatchRandomGain, self).__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        vol = torch.empty(batch_size, device=device)
        vol.uniform_(self.min_gain, self.max_gain)
        mask = torch.rand_like(vol).le(self.p)
        vol[mask] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(batch_size, -1)).view_as(x)
        return vol * x
