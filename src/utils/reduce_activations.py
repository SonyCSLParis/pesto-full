import torch


def reduce_activations(activations: torch.Tensor, reduction: str = "alwa") -> torch.Tensor:
    r"""

    Args:
        activations: tensor of probability activations, shape (batch_size, num_bins)
        reduction (str): reduction method to compute pitch out of activations,
            choose between "argmax", "mean", "alwa".

    Returns:
        torch.Tensor: pitches as fractions of MIDI semitones, shape (batch_size)
    """
    device = activations.device
    num_bins = activations.size(1)
    bps, r = divmod(num_bins, 128)
    assert r == 0, "Activations should have output size 128*bins_per_semitone"

    if reduction == "argmax":
        pred = activations.argmax(dim=1)
        return pred.float() / bps

    all_pitches = torch.arange(num_bins, dtype=torch.float, device=device).div_(bps)
    if reduction == "mean":
        return torch.mm(activations, all_pitches)

    if reduction == "alwa":  # argmax-local weighted averaging, see https://github.com/marl/crepe
        center_bin = activations.argmax(dim=1, keepdim=True)
        window = torch.arange(1, 2 * bps, device=device) - bps
        indices = (window + center_bin).clip_(min=0, max=num_bins - 1)
        cropped_activations = activations.gather(1, indices)
        cropped_pitches = all_pitches.unsqueeze(0).expand_as(activations).gather(1, indices)
        return (cropped_activations * cropped_pitches).sum(dim=1) / cropped_activations.sum(dim=1)

    raise ValueError
