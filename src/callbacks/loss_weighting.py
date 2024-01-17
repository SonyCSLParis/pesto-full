from collections import defaultdict
from math import cos, pi
from typing import Any, Mapping, Optional

import torch

import lightning.pytorch as pl


class LossWeighting(pl.Callback):
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = weights if weights is not None else defaultdict(lambda: 1.)

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Any,
                           batch: Any,
                           batch_idx: int) -> None:
        pl_module.log_dict({f"hparams/{k}_weight": v for k, v in self.weights.items()}, prog_bar=False, logger=True)

    def combine_losses(self, **losses):
        self.update_weights(losses)
        return sum([self.weights[key] * losses[key] for key in self.weights.keys()])

    def update_weights(self, losses):
        pass

    def __str__(self):
        params = '\n'.join(f"\t{k}: {v}" for k, v in vars(self).items())
        return self.__class__.__name__ + "(\n" + params + "\n)"


class WarmupLossWeighting(LossWeighting):
    def __init__(
            self,
            weights: Mapping[str, float],
            warmup_term: str,
            warmup_epochs: int = 10,
            initial_weight: float = 0.,
            warmup_fn: str = "linear"
    ):
        super(WarmupLossWeighting, self).__init__(weights)
        self.key = warmup_term
        self.warmup_epochs = warmup_epochs
        self.initial_weight = initial_weight
        self.target_weight = self.weights[self.key]
        self.warmup_fn = warmup_fn

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch >= self.warmup_epochs:
            self.weights[self.key] = self.target_weight
            return

        # compute new value for the weight
        if self.warmup_fn == "linear":
            weight = (self.target_weight - self.initial_weight) * epoch / self.warmup_epochs + self.initial_weight
        elif self.warmup_fn == "cosine":
            weight = 0.5 * (1 - cos(pi * epoch / self.warmup_epochs)) * (self.target_weight - self.initial_weight) + self.initial_weight
        else:
            raise NotImplementedError(f"This warmup schedule is not supported: `{self.warmup_fn}`.")

        self.weights[self.key] = weight


class GradientsLossWeighting(LossWeighting):
    def __init__(self,
                 weights: Mapping[str, float] | None = None,
                 last_layer: Optional[torch.Tensor] = None,
                 ema_rate: float = 0.):
        super(GradientsLossWeighting, self).__init__(weights)
        self.last_layer = last_layer
        self.ema_rate = ema_rate
        self.grads = {k: 1-v for k, v in weights.items()}
        self.weights_tensor = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.weights_tensor = torch.zeros(len(self.weights.keys()), device=pl_module.device)

    def update_weights(self, losses):
        # compute gradient w.r.t last layer for each loss term
        for i, (k, loss) in enumerate(losses.items()):
            if not loss.requires_grad:
                return

            grads = torch.autograd.grad(loss, self.get_last_layer(k), retain_graph=True)[0].norm().detach()
            old_grads = self.grads[k]
            if old_grads is not None:
                grads = self.ema_rate * old_grads + (1 - self.ema_rate) * grads
            self.grads[k] = grads
            self.weights_tensor[i] = grads

        # compute the weight of this loss based on these gradients
        self.weights_tensor = 1 - self.weights_tensor / self.weights_tensor.sum().clip(min=1e-7)

        # associate each weight with the right loss
        for i, k in enumerate(losses.keys()):
            self.weights[k] = self.weights_tensor[i]

    def get_last_layer(self, key: str) -> torch.Tensor:
        if torch.is_tensor(self.last_layer):
            return self.last_layer
        return self.last_layer[key]

