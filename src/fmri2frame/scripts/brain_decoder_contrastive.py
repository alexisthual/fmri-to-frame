# %%
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fmri2frame.scripts.eval import compute_retrieval_metrics
from fmri2frame.scripts.setup_xp import setup_xp


class BrainDataset(Dataset):
    def __init__(self, brain_features, latent_features):
        self.brain_features = brain_features
        self.latent_features = latent_features
        self.len = len(brain_features)

    def __getitem__(self, index):
        return (
            self.brain_features[index],
            self.latent_features[index],
        )

    def __len__(self):
        return self.len


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

        # Projector
        assert n_proj_blocks >= 0
        projector_layers = []
        for _ in range(n_proj_blocks):
            projector_layers.extend(
                [
                    nn.LayerNorm(hidden_size_projector),
                    nn.GELU(),
                    nn.Linear(hidden_size_projector, hidden_size_projector),
                ]
            )
        projector_layers.extend(
            [
                nn.LayerNorm(hidden_size_projector),
                nn.GELU(),
                nn.Linear(hidden_size_projector, out_dim),
            ]
        )
        self.projector = nn.Sequential(*projector_layers)

    def forward(self, x):
        x = self.lin0(x)

        residual = x
        for res_block in range(self.n_res_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        x = self.lin1(x)

        return x, self.projector(x)


def mixco_symmetrical_nce_loss(
    preds,
    targs,
    temperature=0.1,
    perm=None,
    betas=None,
    select=None,
    bidirectional=True,
):
    """Compute symmetical NCE loss with MixCo augmentation.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted latent features.
    targs : torch.Tensor
        Target latent features.
    temperature : float, optional
        Temperature for the softmax, by default 0.1
    perm : torch.Tensor, optional
        Permutation of the samples, by default None
    betas : torch.Tensor, optional
        Betas for the MixCo augmentation, by default None
    select : torch.Tensor, optional
        Selection of the samples, by default None
    bidirectional : bool, optional
        Whether to compute the loss in both directions, by default True

    Returns
    -------
    torch.Tensor
        Symmetrical NCE loss.
    """
    brain_clip = (preds @ targs.T) / temperature

    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2) / 2
        return loss
    else:
        loss = F.cross_entropy(
            brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device)
        )
        if bidirectional:
            loss2 = F.cross_entropy(
                brain_clip.T,
                torch.arange(brain_clip.shape[0]).to(brain_clip.device),
            )
            loss = (loss + loss2) / 2

        return loss


def mixco_sample_augmentation(samples, beta=0.15, s_thresh=0.5):
    """Augment samples with MixCo augmentation.

    Parameters
    ----------
    samples : torch.Tensor
        Samples to augment.
    beta : float, optional
        Beta parameter for the Beta distribution, by default 0.15
    s_thresh : float, optional
        Proportion of samples which should be affected by MixCo, by default 0.5

    Returns
    -------
    samples : torch.Tensor
        Augmented samples.
    perm : torch.Tensor
        Permutation of the samples.
    betas : torch.Tensor
        Betas for the MixCo augmentation.
    select : torch.Tensor
        Samples affected by MixCo augmentation
    """
    # Randomly select samples to augment
    select = (torch.rand(samples.shape[0]) <= s_thresh).to(samples.device)

    # Randomly select samples used for augmentation
    perm = torch.randperm(samples.shape[0])
    samples_shuffle = samples[perm].to(samples.device, dtype=samples.dtype)

    # Sample MixCo coefficients from a Beta distribution
    betas = (
        torch.distributions.Beta(beta, beta)
        .sample([samples.shape[0]])
        .to(samples.device, dtype=samples.dtype)
    )
    betas[~select] = 1
    betas_shape = [-1] + [1] * (len(samples.shape) - 1)

    # Augment samples
    samples[select] = samples[select] * betas[select].reshape(
        *betas_shape
    ) + samples_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)

    return samples, perm, betas, select


# %%
def train_single_subject_brain_decoder(
    train_dataset_ids=None,
    valid_dataset_ids=None,
    dataset_path=None,
    subject=None,
    lag=0,
    window_size=1,
    latent_type=None,
    pretrained_models_path=None,
    cache=None,
    output_path=None,
    # model + training parameters
    hidden_size_backbone=512,
    hidden_size_projector=512,
    dropout=0.3,
    n_res_blocks=2,
    n_proj_blocks=1,
    temperature=0.01,
    batch_size=128,
    lr=1e-4,
    weight_decay=0,
    n_epochs=50,
):
    # Load training and validation data
    brain_features_train = []
    latent_features_train = []
    brain_features_valid = []
    latent_features_valid = []

    for i, dataset_id in enumerate(train_dataset_ids + valid_dataset_ids):
        datamodule = setup_xp(
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            subject=subject,
            n_train_examples=None,
            n_valid_examples=None,
            n_test_examples=None,
            pretrained_models_path=pretrained_models_path,
            latent_types=[latent_type],
            generation_seed=0,
            batch_size=32,
            window_size=window_size,
            lag=lag,
            agg="mean",
            support="fsaverage5",
            shuffle_labels=False,
            cache=cache,
        )
        if i < len(train_dataset_ids):
            brain_features_train.append(datamodule.train_data.betas)
            latent_features_train.append(datamodule.train_data.labels[latent_type])
        else:
            brain_features_valid.append(datamodule.train_data.betas)
            latent_features_valid.append(datamodule.train_data.labels[latent_type])

    brain_features_train = np.concatenate(brain_features_train)
    latent_features_train = np.concatenate(latent_features_train)
    brain_features_valid = np.concatenate(brain_features_valid)
    latent_features_valid = np.concatenate(latent_features_valid)

    device = torch.device("cuda")
    out_dim = latent_features_train.shape[1]

    # WandB setup
    wandb.login()
    run = wandb.init(
        project=f"inter-species-single-sub-{subject:02d}-sweep",
        dir=Path("/gpfsscratch/rech/nry/uul79xi/wandb"),
        config={
            "out_dim": out_dim,
            "hidden_size_backbone": hidden_size_backbone,
            "hidden_size_projector": hidden_size_projector,
            "dropout": dropout,
            "n_res_blocks": n_res_blocks,
            "n_proj_blocks": n_proj_blocks,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "temperature": temperature,
        },
        save_code=True,
    )

    # Create model
    brain_decoder = BrainDecoder(
        out_dim=out_dim,
        hidden_size_backbone=hidden_size_backbone,
        hidden_size_projector=hidden_size_projector,
        dropout=dropout,
        n_res_blocks=n_res_blocks,
        n_proj_blocks=n_proj_blocks,
    ).to(device)

    n_params = sum([p.numel() for p in brain_decoder.parameters()])
    run.config.update({"n_params": n_params})
    print(f"{n_params:,} parameters")

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in brain_decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in brain_decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        opt_grouped_parameters,
        lr=lr,
    )

    # Create DataLoader
    train_dl = DataLoader(
        BrainDataset(
            brain_features_train,
            latent_features_train,
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        BrainDataset(
            brain_features_valid,
            latent_features_valid,
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Training
    train_mixco_losses = []
    train_symm_nce_losses = []
    lrs = []
    val_mixco_losses = []
    val_symm_nce_losses = []
    val_median_rank = []
    val_topk_acc = defaultdict(list)

    negatives = torch.from_numpy(latent_features_valid).to(device)

    torch.autograd.set_detect_anomaly(True)
    for _ in tqdm(range(n_epochs)):
        # Training step
        brain_decoder.train()
        for train_i, (brain_features, latent_features) in enumerate(train_dl):
            with torch.cuda.amp.autocast():
                brain_features = brain_features.to(device)
                latent_features = latent_features.to(device)

                # Evaluate mixco loss and back-propagate on it
                (
                    brain_features_mixco,
                    perm,
                    betas,
                    select,
                ) = mixco_sample_augmentation(brain_features)

                optimizer.zero_grad()

                _, predicted_latent_features = brain_decoder(brain_features_mixco)

                predicted_latent_features_norm = nn.functional.normalize(
                    predicted_latent_features, dim=-1
                )
                target_latent_features_norm = nn.functional.normalize(
                    latent_features, dim=-1
                )

                mixco_loss = mixco_symmetrical_nce_loss(
                    predicted_latent_features_norm,
                    target_latent_features_norm,
                    temperature=temperature,
                    perm=perm,
                    betas=betas,
                    select=select,
                )

                mixco_loss.backward()
                optimizer.step()

                # Evaluate symmetrical NCE loss
                with torch.no_grad():
                    _, predicted_latent_features = brain_decoder(brain_features)
                    predicted_latent_features_norm = nn.functional.normalize(
                        predicted_latent_features, dim=-1
                    )
                    target_latent_features_norm = nn.functional.normalize(
                        latent_features, dim=-1
                    )
                    symm_nce_loss = mixco_symmetrical_nce_loss(
                        predicted_latent_features_norm,
                        target_latent_features_norm,
                        temperature=temperature,
                    )

            train_mixco_losses.append(mixco_loss.item())
            train_symm_nce_losses.append(symm_nce_loss.item())
            lrs.append(optimizer.param_groups[-1]["lr"])

        # Validation step
        brain_decoder.eval()
        epoch_val_mixco_losses = []
        epoch_val_symm_nce_losses = []
        epoch_median_rank = []
        epoch_topk_acc = defaultdict(list)
        epoch_ranks = []
        top_k = [1, 5, 10, int(len(negatives) / 10), int(len(negatives) / 5)]

        for val_i, (brain_features, latent_features) in enumerate(val_dl):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    brain_features = brain_features.to(device)
                    latent_features = latent_features.to(device)

                    _, predicted_latent_features = brain_decoder(brain_features)

                    # Evaluate retrieval metrics
                    retrieval_metrics = compute_retrieval_metrics(
                        predicted_latent_features,
                        latent_features,
                        negatives,
                        return_scores=True,
                        top_k=top_k,
                    )
                    epoch_median_rank.append(retrieval_metrics["relative_median_rank"])
                    for k in top_k:
                        epoch_topk_acc[k].append(retrieval_metrics[f"top-{k}_accuracy"])
                    all_scores = retrieval_metrics["scores"].cpu().numpy()
                    ranks = (all_scores > all_scores[:, [0]]).sum(axis=1)
                    epoch_ranks.extend(ranks)

                    # Evaluate symmetrical NCE loss
                    predicted_latent_features_norm = nn.functional.normalize(
                        predicted_latent_features, dim=-1
                    )
                    target_latent_features_norm = nn.functional.normalize(
                        latent_features, dim=-1
                    )
                    symm_nce_loss = mixco_symmetrical_nce_loss(
                        predicted_latent_features_norm,
                        target_latent_features_norm,
                        temperature=temperature,
                    )
                    epoch_val_symm_nce_losses.append(symm_nce_loss.item())

                    # Evaluate mixco symmetrical NCE loss
                    (
                        brain_features_mixco,
                        perm,
                        betas,
                        select,
                    ) = mixco_sample_augmentation(brain_features)

                    _, predicted_latent_features = brain_decoder(brain_features_mixco)

                    predicted_latent_features_norm = nn.functional.normalize(
                        predicted_latent_features, dim=-1
                    )
                    target_latent_features_norm = nn.functional.normalize(
                        latent_features, dim=-1
                    )

                    mixco_loss = mixco_symmetrical_nce_loss(
                        predicted_latent_features_norm,
                        target_latent_features_norm,
                        temperature=temperature,
                        perm=perm,
                        betas=betas,
                        select=select,
                    )
                    epoch_val_mixco_losses.append(mixco_loss.item())

        val_mixco_losses.append(epoch_val_mixco_losses)
        val_symm_nce_losses.append(epoch_val_symm_nce_losses)
        val_median_rank.append(epoch_median_rank)
        for k in top_k:
            val_topk_acc[k].append(epoch_topk_acc[k])

        run.config.update({"n_epochs": len(val_mixco_losses)})
        run.config.update({"n_retrieval_set": len(negatives)})

        wandb.log(
            {
                **{
                    "train/mixco_loss": np.mean(train_mixco_losses[-1]),
                    "train/symm_nce_loss": np.mean(train_symm_nce_losses[-1]),
                    "train/lr": lrs[-1],
                    "val/mixco_loss": np.mean(val_mixco_losses[-1]),
                    "val/symm_nce_loss": np.mean(val_symm_nce_losses[-1]),
                    "val/median_rank": np.mean(val_median_rank[-1]),
                    "val/ranks": wandb.Histogram(np.array(epoch_ranks), num_bins=100),
                },
                **{f"val/top-{k}_acc": np.mean(val_topk_acc[k][-1]) for k in top_k},
            }
        )
