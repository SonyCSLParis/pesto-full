import torch
import torch.nn as nn


def randint_sampling_fn(min_value, max_value):
    def sample_randint(*size, **kwargs):
        return torch.randint(min_value, max_value+1, size, **kwargs)

    return sample_randint


def gaussint_sampling_fn(min_value, max_value):
    mean = (min_value + max_value) / 2
    std = (max_value - mean) / 2
    def sample_gaussint(*size, **kwargs):
        return torch.randn(size, **kwargs).add_(mean).mul_(std).long().clip(min=min_value, max=max_value)
    return sample_gaussint


class PitchShiftCQT(nn.Module):
    def __init__(self,
                 min_steps: int,
                 max_steps: int,
                 gaussian_sampling: bool = False):
        super(PitchShiftCQT, self).__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.sample_random_steps = gaussint_sampling_fn(min_steps, max_steps) if gaussian_sampling \
            else randint_sampling_fn(min_steps, max_steps)

        # lower bin
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor):
        batch_size, _, input_height = spectrograms.size()
        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0, \
            f"With input height {input_height:d} and output height {output_height:d}, impossible " \
            f"to have a range of {self.max_steps - self.min_steps:d} bins."

        n_steps = self.sample_random_steps(batch_size, device=spectrograms.device)
        x = spectrograms[..., self.lower_bin: self.lower_bin + output_height]
        xt = self.extract_bins(spectrograms, self.lower_bin - n_steps, output_height)

        return x, xt, n_steps

    def extract_bins(self, inputs: torch.Tensor, first_bin: torch.LongTensor, output_height: int):
        r"""Efficiently extract subsegments of CQT of size `output_height`,
        i.e. so that outputs[i, j] = inputs[i, ..., first_bin[j] : first_bin[j] + self.output_height]

        Args:
            inputs (torch.Tensor): tensor of CQTs, shape (batch_size, num_channels, input_height)
            first_bin (torch.LongTensor): indices of the first bin of each segment, shape (batch_size)
            output_height (int): output height of the cropped CQT

        Returns:
            segments of CQTs, shape (batch_size, num_channels, output_height)
        """
        indices = first_bin.unsqueeze(-1) + torch.arange(output_height, device=inputs.device)
        dims = inputs.size(0), 1, output_height

        output_size = list(inputs.size())[:-1] + [output_height]
        indices = indices.view(*dims).expand(output_size)
        return inputs.gather(-1, indices)
