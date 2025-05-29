import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(
        self, time_mask_param=30, freq_mask_param=13, num_masks=2, fill_value=0.0, apply_prob=1.0
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.fill_value = fill_value
        self.apply_prob = apply_prob

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec (Tensor): shape (1, n_mels, time_steps) or (batch, 1, n_mels, time_steps)
        Returns:
            Tensor: augmented spectrogram
        """
        if not self.training or random.random() > self.apply_prob:
            return spec

        # Allow both batched and unbatched inputs
        batched = spec.dim() == 4
        if not batched:
            spec = spec.unsqueeze(0)

        spec = spec.clone()

        for s in spec:
            # Frequency masking
            for _ in range(self.num_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, s.shape[1] - f)
                s[0, f0 : f0 + f, :] = self.fill_value

            # Time masking
            for _ in range(self.num_masks):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, s.shape[2] - t)
                s[0, :, t0 : t0 + t] = self.fill_value

        return spec if batched else spec.squeeze(0)
