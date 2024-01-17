import hashlib
import itertools
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import torchaudio
from lightning import LightningDataModule

from src.data.hcqt import HarmonicCQT


log = logging.getLogger(__name__)


def hz_to_mid(freqs):
    return np.where(freqs > 0, 12 * np.log2(freqs / 440) + 69, 0)


class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels=None, filter_unvoiced: bool = False) -> None:
        assert labels is None or len(inputs) == len(labels), "Lengths of inputs and labels do not match"
        if filter_unvoiced and labels is None:
            log.warning("Cannnot filter out unvoiced frames without annotations.")
            filter_unvoiced = False
        if filter_unvoiced:
            self.inputs = inputs[labels > 0]
            self.labels = labels[labels > 0]
        else:
            self.inputs = inputs
            self.labels = labels

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[item] if self.labels is not None else 0
        return torch.view_as_complex(torch.from_numpy(self.inputs[item])), label

    def __len__(self):
        return len(self.inputs)


class AudioDataModule(LightningDataModule):
    def __init__(self,
                 audio_files: str,
                 annot_files: str | None = None,
                 val_audio_files: str | None = None,
                 val_annot_files: str | None = None,
                 harmonics: Sequence[float] = (1,),
                 hop_duration: float = 10.,
                 fmin: float = 27.5,
                 fmax: float | None = None,
                 bins_per_semitone: int = 1,
                 n_bins: int = 84,
                 center_bins: bool = False,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 transforms: Sequence[torch.nn.Module] | None = None,
                 fold: int | None = None,
                 n_folds: int = 5,
                 cache_dir: str = "./cache",
                 filter_unvoiced: bool = False,
                 mmap_mode: str | None = None):
        r"""

        Args:
            audio_files: path to csv file containing the list of audio files to process

        """
        super(AudioDataModule, self).__init__()

        # sanity checks
        assert val_audio_files is None or val_annot_files is not None, "Validation set (if it exists) must be annotated"
        assert val_audio_files is None or fold is None, "Specify `val_audio_files` OR cross-validation `fold`, not both"
        assert annot_files is not None or fold is None, "Cannot perform cross-validation without any annotations."

        self.audio_files = Path(audio_files)
        self.annot_files = Path(annot_files) if annot_files is not None else None

        if val_audio_files is not None:
            self.val_audio_files = Path(val_audio_files)
            self.val_annot_files = Path(val_annot_files)
        else:
            self.val_audio_files = None
            self.val_annot_files = None

        self.fold = fold
        self.n_folds = n_folds

        # HCQT
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_duration = hop_duration

        self.hcqt_kwargs = dict(
            harmonics=list(harmonics),
            fmin=fmin,
            fmax=fmax,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins
        )

        # dataloader keyword-arguments
        self.dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # transforms
        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()

        # misc
        self.cache_dir = Path(cache_dir)
        self.filter_unvoiced = filter_unvoiced
        self.mmap_mode = mmap_mode

        # placeholders for datasets and samplers
        self.train_dataset = None
        self.train_sampler = None
        self.val_dataset = None
        self.val_sampler = None

    def prepare_data(self) -> None:
        self.train_dataset = self.load_data(self.audio_files, self.annot_files)
        if self.val_audio_files is not None:
            self.val_dataset = self.load_data(self.val_audio_files, self.val_annot_files)

    def setup(self, stage: str) -> None:
        # If the dataset is labeled, we split it randomly and keep 20% for validation only
        # Otherwise we train on the whole dataset
        if self.val_dataset is not None:
            return

        if not self.annot_files:
            # create dummy validation set
            self.val_dataset = NpyDataset(np.zeros_like(self.train_dataset.inputs[:1]))
            return

        self.val_dataset = self.load_data(self.audio_files, self.annot_files)

        if self.fold is not None:
            # see https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
            from sklearn.model_selection import KFold

            # We fix random_state=0 for the train/val split to be consistent across runs, even if the global seed changes
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
            iterator = kfold.split(self.train_dataset)
            train_idx, val_idx = None, None  # just to make the linter shut up
            for _ in range(self.fold + 1):
                train_idx, val_idx = next(iterator)

            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, sampler=self.val_sampler, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x, y = batch
        return self.transforms(x), y

    def load_data(self, audio_files: Path, annot_files: Path | None = None) -> torch.utils.data.Dataset:
        cache_cqt = self.build_cqt_filename(audio_files)
        if cache_cqt.exists():
            inputs = np.load(cache_cqt, mmap_mode=self.mmap_mode)
            cache_annot = cache_cqt.with_suffix(".csv")
            annotations = np.loadtxt(cache_annot, dtype=np.float32) if cache_annot.exists() else None
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            inputs, annotations = self.precompute_hcqt(audio_files, annot_files)
            np.save(cache_cqt, inputs, allow_pickle=False)
            if annotations is not None:
                np.savetxt(cache_cqt.with_suffix(".csv"), annotations)
        return NpyDataset(inputs, labels=annotations, filter_unvoiced=self.filter_unvoiced)

    def build_cqt_filename(self, audio_files) -> Path:
        # build a hash
        dict_str = json.dumps({
            "audio_files": str(audio_files),
            "hop_duration": self.hop_duration,
            **self.hcqt_kwargs
        }, sort_keys=True)
        hash_id = hashlib.sha256(dict_str.encode()).hexdigest()[:8]

        # build filename
        fname = "hcqt_" + hash_id + ".npy"
        return self.cache_dir / fname

    def precompute_hcqt(self, audio_path: Path, annot_path: Path | None = None) -> Tuple[np.ndarray,np.ndarray]:
        data_dir = audio_path.parent

        cqt_list = []
        with audio_path.open('r') as f:
            audio_files = f.readlines()

        if annot_path is not None:
            with annot_path.open('r') as f:
                annot_files = f.readlines()
            annot_list = []
        else:
            annot_files = []
            annot_list = None

        log.info("Precomputing HCQT...")
        pbar = tqdm(itertools.zip_longest(audio_files, annot_files, fillvalue=None),
                    total=len(audio_files),
                    leave=False)
        for fname, annot in pbar:
            fname = fname.strip()
            pbar.set_description(fname)
            x, sr = torchaudio.load(data_dir / fname)
            out = self.hcqt(x.mean(dim=0), sr)  # convert to mono and compute HCQT

            if annot is not None:
                annot = annot.strip()
                timesteps, freqs = np.loadtxt(data_dir / annot, delimiter=',', dtype=np.float32).T
                hop_duration = 1000 * (timesteps[1] - timesteps[0])

                # Badly-aligned annotations is a fucking nightmare
                # so we double-check for each file that hop sizes and lengths do match.
                # Since hop sizes are floats we put a tolerance of 1e-6 in the equality
                assert abs(hop_duration - self.hop_duration) < 1e-6, \
                    (f"Inconsistency between {fname} and {annot}:\n"
                     f"the resolution of the annotations ({len(freqs):d}) "
                     f"does not match the number of CQT frames ({len(out):d}). "
                     f"The hop duration between CQT frames should be identical "
                     f"but got {hop_duration:.1f} ms vs {self.hop_duration:.1f} ms. "
                     f"Please either adjust the hop duration of the CQT or resample the annotations.")
                assert len(out) == len(freqs), \
                    (f"Inconsistency between {fname} and {annot}:"
                     f"the resolution of the annotations ({len(freqs):d}) "
                     f"does not match the number of CQT frames ({len(out):d}) "
                     f"despite hop durations match. "
                     f"Please check that your annotations are correct.")
                annot_list.append(hz_to_mid(freqs))

            cqt_list.append(out.cpu().numpy())

        return np.concatenate(cqt_list), np.concatenate(annot_list) if annot_list is not None else None

    def hcqt(self, audio: torch.Tensor, sr: int):
        # compute CQT kernels if it does not exist yet
        if sr != self.hcqt_sr:
            self.hcqt_sr = sr
            hop_length = int(self.hop_duration * sr / 1000 + 0.5)
            self.hcqt_kernels = HarmonicCQT(sr=sr, hop_length=hop_length, **self.hcqt_kwargs)

        return self.hcqt_kernels(audio).squeeze(0).permute(2, 0, 1, 3)  # (time, harmonics, freq_bins, 2)
