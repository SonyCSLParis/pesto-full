import logging
from typing import Mapping

import numpy as np
import torch

import mir_eval.melody as mir_eval

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WANDB_AVAILABLE = False


log = logging.getLogger(__name__)


def wandb_only(func):
    def wrapper(*args, **kwargs):
        if WANDB_AVAILABLE:
            return func(*args, **kwargs)
        log.warning(f"Method {func.__name__} can be used only with wandb.")
        return None
    return wrapper


class MIREvalCallback(pl.Callback):
    def __init__(self,
                 bins_per_semitone: int = 1,
                 reduction: str = "alwa",
                 cdf_resolution: int = 0):
        super(MIREvalCallback, self).__init__()
        self.bps = bins_per_semitone
        self.reduction = reduction

        self.logger = None

        self.cdf_resolution = cdf_resolution

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
                break
        if self.logger is None:
            global WANDB_AVAILABLE
            WANDB_AVAILABLE = False
            log.warning(f"As of now, `{self.__class__.__name__}` is only compatible with `WandbLogger`. "
                             f"Loggers: {pl_module.loggers}.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        predictions = torch.cat(pl_module.predictions).cpu().numpy()
        labels = torch.cat(pl_module.labels).cpu().numpy()
        log_path = "accuracy/{}"

        metrics = self.compute_metrics(predictions, labels)
        pl_module.log_dict({log_path.format(k): v for k, v in metrics.items()}, sync_dist=True)

        self.plot_pitch_error_cdf(predictions, labels, labels > 0)

    @staticmethod
    def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Mapping[str, float]:
        # convert semitones to cents and infer voicing
        ref_cent, ref_voicing = mir_eval.freq_to_voicing(100 * labels)
        est_cent, est_voicing = mir_eval.freq_to_voicing(100 * predictions)

        # compute mir_eval metrics
        metrics = {}
        metrics["RPA"] = mir_eval.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
        metrics["RCA"] = mir_eval.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
        metrics["OA"] = mir_eval.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)

        return metrics

    @wandb_only
    def plot_pitch_error_cdf(self, predictions: np.ndarray, labels: np.ndarray, voiced: np.ndarray):
        sorted_errors = np.sort(np.abs(predictions[voiced] - labels[voiced]))
        total = len(sorted_errors)
        cumul_probs = np.arange(1, total + 1) / total

        cols = ["Pitch error (semitones)", "Cumulative Density Function"]
        fig = wandb.Table(data=list(zip(sorted_errors[::self.cdf_resolution], cumul_probs[::self.cdf_resolution])),
                          columns=cols)
        self.logger.experiment.log({"pitch_error": wandb.plot.line(fig, *cols)})
