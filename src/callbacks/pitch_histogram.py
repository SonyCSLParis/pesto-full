import logging
import numpy as np
import torch

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


class PitchHistogramCallback(pl.Callback):
    def __init__(self):
        super(PitchHistogramCallback, self).__init__()
        self.logger = None

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
        predictions = torch.cat(pl_module.predictions)
        self.plot_pitch_histogram(predictions + pl_module.shift)  # we unshift distributions there to see better what's going on

    @wandb_only
    def plot_pitch_histogram(self, predictions: np.ndarray):
        fig = wandb.Table(data=[[p] for p in predictions], columns=["predictions"])
        self.logger.experiment.log({"pitch_histogram": wandb.plot.histogram(fig,
                                                                            "predictions",
                                                                            title="Pitch histogram")})
