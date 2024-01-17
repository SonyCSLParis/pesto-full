import torch
import torch.nn as nn

from nnAudio.features.cqt import CQT


class HarmonicCQT(nn.Module):
    def __init__(
            self,
            harmonics,
            sr: int = 22050,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: float | None = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True
    ):
        super(HarmonicCQT, self).__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList([
            CQT(sr=sr, hop_length=hop_length, fmin=h*fmin, fmax=fmax, n_bins=n_bins,
                bins_per_octave=12*bins_per_semitone, output_format="Complex", verbose=False)
            for h in harmonics
        ])

    def forward(self, audio_waveforms: torch.Tensor):
        r"""

        Returns:
            Harmonic CQT, shape (num_channels, num_harmonics, num_freqs, num_timesteps, 2)
        """
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)
