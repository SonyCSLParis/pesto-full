import abc
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class Loss(_Loss, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass


class ComposeLoss(Loss):
    def __init__(self, losses: Dict[str, Loss], input_dims: Dict[str, int]):
        super(ComposeLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        self.input_dims = [input_dims[k] for k in self.losses.keys()]

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        chunks = inputs.split(self.input_dims, dim=-1)
        loss_dict = {}

        total_loss = None

        for (k, loss_fn), chunk in zip(self.losses.items(), chunks):
            print(k, loss_fn, chunk.size())
            # compute loss
            aux_dict = loss_fn(chunk)

            # retrieve total auxiliary loss
            loss = aux_dict.pop("loss")

            # add it to total loss
            if total_loss is None:
                total_loss = loss.clone()
            else:
                total_loss = total_loss + loss

            # write quantities into dict
            loss_dict.update(aux_dict)
            loss_dict[k] = loss

        loss_dict["loss"] = total_loss
        print(loss_dict)
        return loss_dict


class NullLoss(nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return args[0].mean().mul(0)
