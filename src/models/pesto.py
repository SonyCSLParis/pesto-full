import logging
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch
import torch.nn as nn
from lightning import LightningModule

from src.callbacks.loss_weighting import LossWeighting
from src.data.pitch_shift import PitchShiftCQT
from src.losses import NullLoss
from src.utils import reduce_activations, remove_omegaconf_dependencies
from src.utils.calibration import generate_synth_data


log = logging.getLogger(__name__)


class PESTO(LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 equiv_loss_fn: nn.Module | None = None,
                 sce_loss_fn: nn.Module | None = None,
                 inv_loss_fn: nn.Module | None = None,
                 pitch_shift_kwargs: Mapping[str, Any] | None = None,
                 transforms: Sequence[nn.Module] | None = None,
                 reduction: str = "alwa"):
        super(PESTO, self).__init__()
        self.encoder = encoder
        self.optimizer_cls = optimizer
        self.scheduler_cls = scheduler

        # loss definitions
        self.equiv_loss_fn = equiv_loss_fn or NullLoss()
        self.sce_loss_fn = sce_loss_fn or NullLoss()
        self.inv_loss_fn = inv_loss_fn or NullLoss()

        # pitch-shift CQT
        if pitch_shift_kwargs is None:
            pitch_shift_kwargs = {}
        self.pitch_shift = PitchShiftCQT(**pitch_shift_kwargs)

        # preprocessing and transforms
        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()
        self.reduction = reduction

        # loss weighting
        self.loss_weighting = None

        # predictions and labels
        self.predictions = None
        self.labels = None

        # constant shift to get absolute pitch from predictions
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

        # save hparams
        self.hyperparams = dict(encoder=encoder.hparams, pitch_shift=pitch_shift_kwargs)

    def forward(self,
                x: torch.Tensor,
                shift: bool = True,
                return_activations: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, *_ = self.pitch_shift(x)  # the CQT has to be cropped beforehand

        activations = self.encoder(x)
        preds = reduce_activations(activations, reduction=self.reduction)

        if shift:
            preds.sub_(self.shift)

        if return_activations:
            return preds, activations

        return preds

    def on_fit_start(self) -> None:
        r"""Search among Trainer's checkpoints if there is a `LossWeighting`.
        If so, then identify it to use it for training.
        Otherwise create a dummy one.
        """
        for callback in self.trainer.callbacks:
            if isinstance(callback, LossWeighting):
                self.loss_weighting = callback
        if self.loss_weighting is None:
            self.loss_weighting = LossWeighting()
        self.loss_weighting.last_layer = self.encoder.fc.weight

    def on_validation_epoch_start(self) -> None:
        self.predictions = []
        self.labels = []

        self.estimate_shift()

    def on_validation_batch_end(self,
                                outputs,
                                batch,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        preds, labels = outputs
        self.predictions.append(preds)
        self.labels.append(labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch  # we do not use the eventual labels during training

        # pitch-shift
        x, xt, n_steps = self.pitch_shift(x)
        xa = x.clone()

        xa = self.transforms(xa)
        xt = self.transforms(xt)

        # pass through network
        y = self.encoder(x)
        ya = self.encoder(xa)
        yt = self.encoder(xt)

        # invariance
        inv_loss = self.inv_loss_fn(y, ya)

        # shift-entropy
        shift_entropy_loss = self.sce_loss_fn(ya, yt, n_steps)

        # equivariance
        equiv_loss = self.equiv_loss_fn(ya, yt, n_steps)  # WARNING: augmented view is y2t!

        # weighting
        total_loss = self.loss_weighting.combine_losses(invariance=inv_loss,
                                                        shift_entropy=shift_entropy_loss,
                                                        equivariance=equiv_loss)

        # add elems to dict
        loss_dict = dict(invariance=inv_loss,
                         equivariance=equiv_loss,
                         shift_entropy=shift_entropy_loss,
                         loss=total_loss)

        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, pitch = batch
        return self.forward(x), pitch

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""We store the hyperparameters of the encoder for inference from outside.
        It is not used in this repo but enables to load the model from the pip-installable inference repository.
        """
        checkpoint["hparams"] = remove_omegaconf_dependencies(self.hyperparams)
        checkpoint['hcqt_params'] = remove_omegaconf_dependencies(self.trainer.datamodule.hcqt_kwargs)

    def configure_optimizers(self) -> Mapping[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer_cls(params=self.encoder.parameters())
        monitor = dict(optimizer=optimizer)
        if self.scheduler_cls is not None:
            monitor["lr_scheduler"] = self.scheduler_cls(optimizer=optimizer)

        return monitor

    def estimate_shift(self) -> None:
        r"""Estimate the shift to predict absolute pitches from relative activations"""
        # 0. Define labels
        labels = torch.arange(60, 72)

        # 1. Generate synthetic audio and convert it to HCQT
        sr = 16000
        dm = self.trainer.datamodule
        batch = []
        for p in labels:
            audio = generate_synth_data(p, sr=sr)
            hcqt = dm.hcqt(audio, sr)
            batch.append(hcqt[0])

        # 2. Stack batch and apply final transforms
        x = torch.stack(batch, dim=0).to(self.device)
        x = dm.transforms(torch.view_as_complex(x))

        # 3. Pass it through the module
        preds = self.forward(x, shift=False)

        # 4. Compute the difference between the predictions and the expected values
        diff = preds - labels.to(self.device)

        # 5. Define the shift as the median distance and check that the std is low-enough
        shift, std = diff.median(), diff.std()

        log.info(f"Estimated shift: {shift.cpu().item():.3f} (std = {std.cpu().item():.3f})")

        # 6. Update `self.shift` value
        self.shift.fill_(shift)
