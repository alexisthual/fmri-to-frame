from functools import partial

import torch.nn as nn


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dim=10242 * 2,
        out_dim=768,
        hidden_size_backbone=512,
        hidden_size_projector=512,
        dropout=0.2,
        n_res_blocks=2,
        n_proj_blocks=1,
        norm_type="ln",
        activation_layer_first=False,
    ):
        super().__init__()

        self.n_res_blocks = n_res_blocks

        norm_backbone = (
            partial(nn.BatchNorm1d, num_features=hidden_size_backbone)
            if norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=hidden_size_backbone)
        )
        activation_backbone = (
            partial(nn.ReLU, inplace=True) if norm_type == "bn" else nn.GELU
        )
        activation_and_norm = (
            (activation_backbone, norm_backbone)
            if activation_layer_first
            else (norm_backbone, activation_backbone)
        )

        # First linear
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, hidden_size_backbone),
            *[item() for item in activation_and_norm],
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size_backbone, hidden_size_backbone),
                    *[item() for item in activation_and_norm],
                    nn.Dropout(dropout),
                )
                for _ in range(n_res_blocks)
            ]
        )

        # Second linear
        self.lin1 = nn.Linear(hidden_size_backbone, hidden_size_projector, bias=True)

        assert n_proj_blocks >= 0

        # Contrastive projector
        projector_layers_contrastive = []
        for _ in range(n_proj_blocks):
            projector_layers_contrastive.extend(
                [
                    nn.LayerNorm(hidden_size_projector),
                    nn.GELU(),
                    nn.Linear(hidden_size_projector, hidden_size_projector),
                ]
            )
        projector_layers_contrastive.extend(
            [
                nn.LayerNorm(hidden_size_projector),
                nn.GELU(),
                nn.Linear(hidden_size_projector, out_dim),
            ]
        )
        self.projector_contrastive = nn.Sequential(*projector_layers_contrastive)

        # Reconstruction projector
        # projector_layers_reconstruction = []
        # for _ in range(n_proj_blocks):
        #     projector_layers_reconstruction.extend(
        #         [
        #             nn.LayerNorm(hidden_size_projector),
        #             nn.GELU(),
        #             nn.Linear(hidden_size_projector, hidden_size_projector),
        #         ]
        #     )
        # projector_layers_reconstruction.extend(
        #     [
        #         nn.LayerNorm(hidden_size_projector),
        #         nn.GELU(),
        #         nn.Linear(hidden_size_projector, out_dim),
        #     ]
        # )
        # self.projector_reconstruction = nn.Sequential(*projector_layers_reconstruction)

    def forward(self, x):
        x = self.lin0(x)

        residual = x
        for res_block in range(self.n_res_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        x = self.lin1(x)

        # return x, self.projector_contrastive(x), self.projector_reconstruction(x)
        return x, self.projector_contrastive(x)
