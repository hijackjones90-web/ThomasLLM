"""MLP projector mapping vision features into the language model embedding space."""
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """2-layer GELU MLP: vision_dim -> hidden_dim -> text_dim.

    The second layer bias is zero-initialised so the initial projection is a
    neutral zero-ish offset, which stabilises the first alignment steps.
    """

    def __init__(self, vision_dim: int = 1152, text_dim: int = 1024, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, text_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """vision_features: [B, num_patches, vision_dim] -> [B, num_patches, text_dim]."""
        return self.fc2(self.act(self.fc1(vision_features)))
