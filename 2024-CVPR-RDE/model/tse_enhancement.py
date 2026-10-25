"""Optional, modality-specific enhancement of RDE's TSE features."""

import math
import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """Return a small feature update; the caller applies the residual."""

    def __init__(self, feature_dim, bottleneck_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        # A small nonzero weight keeps the initial mapping close to identity
        # while allowing gradients through the entire adapter on step one.
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        return self.up(self.activation(self.down(self.norm(feature))))


class TSEEnhancer(nn.Module):
    """Enhance one modality without mixing image and text samples."""

    def __init__(self, mode, global_dim, tse_dim, bottleneck_dim=256, scale=0.1):
        super().__init__()
        if mode not in ("adapter", "delta_gate"):
            raise ValueError("TSEEnhancer mode must be adapter or delta_gate")
        if global_dim < 1 or tse_dim < 1 or bottleneck_dim < 1:
            raise ValueError("feature and bottleneck dimensions must be positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("adapter scale must be finite and positive")

        self.mode = mode
        self.scale = scale
        self.adapter = ResidualAdapter(tse_dim, bottleneck_dim)
        if mode == "delta_gate":
            self.global_projection = nn.Linear(global_dim, tse_dim)
            self.gate = nn.Sequential(
                nn.LayerNorm(3 * tse_dim),
                nn.Linear(3 * tse_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, tse_dim),
                nn.Sigmoid(),
            )
            nn.init.normal_(self.gate[3].weight, std=1e-3)
            nn.init.zeros_(self.gate[3].bias)

    def forward(self, tse_feature, global_feature=None):
        # Original RDE returns float32 BGE/TSE features. Keep the new MLP in
        # float32 even though CLIP and original TSE Linear layers use fp16.
        with torch.cuda.amp.autocast(enabled=False):
            tse_feature = tse_feature.float()
            update = self.adapter(tse_feature)
            if self.mode == "adapter":
                return tse_feature + self.scale * update
            if global_feature is None:
                raise ValueError("delta_gate requires the same-modality BGE feature")
            projected = self.global_projection(global_feature.float())
            gate_input = torch.cat((projected, tse_feature,
                                    torch.abs(projected - tse_feature)), dim=-1)
            gate = self.gate(gate_input)
            return tse_feature + self.scale * gate * update
