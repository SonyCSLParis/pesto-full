from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 symmetric: bool = False,
                 detach_targets: bool = False,
                 backend: nn.Module = nn.CrossEntropyLoss()) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.symmetric = symmetric
        self.detach_targets = detach_targets
        self.backend = backend

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.symmetric:
            return (self.compute_loss(input, target) + self.compute_loss(target, input)) / 2
        return self.compute_loss(input, target)

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.backend(input, target.detach() if self.detach_targets else target)


class ShiftCrossEntropy(nn.Module):
    def __init__(self,
                 pad_length: int = 5,
                 criterion: nn.Module = CrossEntropyLoss()):
        super(ShiftCrossEntropy, self).__init__()
        self.criterion = criterion
        self.pad_length = pad_length

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""x2[i] is the pitch-shifted version of x1[i] by target[i] semitones, i.e.
            if x1[i] is a C# and x2[i] is a C then target[i] = -1

        """
        # pad x1 and x2
        x1 = F.pad(x1, (self.pad_length, self.pad_length))
        x2 = F.pad(x2, (2*self.pad_length, 2*self.pad_length))

        # shift x2
        idx = target.unsqueeze(1) + torch.arange(x1.size(-1), device=target.device) + self.pad_length
        shift_x2 = torch.gather(x2, dim=1, index=idx)

        # compute loss
        return self.criterion(x1, shift_x2)
