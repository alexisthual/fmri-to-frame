import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fugw.utils import load_mapping
from scipy.linalg import norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from fmri2frame.scripts.brain_decoder_contrastive import BrainDecoder
from fmri2frame.scripts.compute_alignment import get_n_vertices_left
from fmri2frame.scripts.generate_captions import generate_captions
from fmri2frame.scripts.setup_xp import setup_xp


def pearson_r(a, b):
    """Compute Pearson correlation between x and y.

    Compute Pearson correlation between 2d arrays x and y
    along the samples axis.
    Adapted from scipy.stats.pearsonr.

    Parameters
    ----------
    a: np.ndarray or torch.Tensor of size (n_samples, n_features)
    b: np.ndarray or torch.Tensor of size (n_samples, n_features)

    Returns
    -------
    r: np.ndarray of size (n_samples,)
    """
    if torch.is_tensor(a):
        x = a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        x = a
    elif isinstance(a, list):
        x = np.array(a)
    else:
        raise ValueError("a must be a list, np.ndarray or torch.Tensor")

    if torch.is_tensor(b):
        y = b.detach().cpu().numpy()
    elif isinstance(b, np.ndarray):
        y = b
    elif isinstance(b, list):
        y = np.array(b)
    else:
        raise ValueError("b must be a list, np.ndarray or torch.Tensor")

    dtype = type(1.0 + x[0, 0] + y[0, 0])

    xmean = x.mean(axis=1, dtype=dtype)
    ymean = y.mean(axis=1, dtype=dtype)

    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean[:, np.newaxis]
    ym = y.astype(dtype) - ymean[:, np.newaxis]

    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = norm(xm, axis=1)
    normym = norm(ym, axis=1)

    r = np.sum((xm / normxm[:, np.newaxis]) * (ym / normym[:, np.newaxis]), axis=1)

    return r


def compute_retrieval_metrics(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    negatives: torch.Tensor,
    should_plot: bool = False,
    return_scores: bool = False,
    top_k=[1, 5, 10],
) -> dict:
    """Compute retrieval metrics."""
    # Compute score between predictions and ground truth latents: size (b,)
    outputs = {}
    n_retrieval_set = negatives.shape[0] + 1

    outputs.update({"n_retrieval_set": n_retrieval_set})

    inv_norms_ground_truth = 1 / (
        1e-16 + ground_truth.to(torch.float32).norm(dim=1, p=2)
    )
    scores_to_ground_truth = torch.einsum(
        "sn,sn,s->s",
        predictions.to(torch.float32),
        ground_truth.to(torch.float32),
        inv_norms_ground_truth,
    )
    # Compute score between predictions and other latents: size (b, n)
    inv_norms_negatives = 1 / (1e-16 + negatives.to(torch.float32).norm(dim=1, p=2))
    scores_to_others = torch.einsum(
        "sn,tn,t->st",
        predictions.to(torch.float32),
        negatives.to(torch.float32),
        inv_norms_negatives,
    )

    # Concat scores: size (b, n+1)
    all_scores = torch.cat(
        [scores_to_ground_truth.reshape(-1, 1), scores_to_others], dim=1
    )

    ranks = (all_scores > all_scores[:, [0]]).sum(axis=1)

    # Median rank of ground truth is that of the first column
    median_rank = torch.median(ranks)
    relative_median_rank = median_rank / n_retrieval_set

    if should_plot:
        n, _, _ = plt.hist(ranks.numpy(), bins=100)
        plt.vlines(median_rank.item(), 0, np.max(n), color="black")
        plt.vlines(
            n_retrieval_set / 2,
            0,
            np.max(n),
            color="black",
            linestyles="dotted",
        )
        plt.show()

    outputs.update({"relative_median_rank": relative_median_rank.item()})

    # n-way top-k accuracies
    for k in top_k:
        # It can be that the current batch is smaller than k,
        # in which case topk would raise an error.
        min_k = min(k, all_scores.shape[1])
        accuracy = (ranks < min_k).sum() / ranks.shape[0]
        outputs.update({f"top-{k}_accuracy": accuracy.item()})

    if return_scores:
        outputs.update({"scores": all_scores})

    return outputs


def evaluate_brain_decoder(
    decoder_path=None,
    decoder_is_contrastive=False,
    dataset_ids=None,
    dataset_path=None,
    subject=None,
    subject_is_macaque=False,
    lag=0,
    window_size=1,
    latent_type=None,
    pretrained_models_path=None,
    cache=None,
    left_mapping_path=None,
    right_mapping_path=None,
    invert_mapping=False,
    selected_indices_left_path=None,
    selected_indices_right_path=None,
    output_name=None,
    output_path=None,
    should_generate_captions=True,
):
    """Evaluate brain decoder."""
    device = torch.device("cuda")

    # Load model
    if decoder_is_contrastive:
        checkpoint = torch.load(decoder_path, map_location="cpu")
        # brain_decoder_params = {
        #     "out_dim": 768,
        #     "hidden_size_backbone": 512,
        #     "hidden_size_projector": 512,
        #     "dropout": 0.8,
        #     "n_res_blocks": 2,
        #     "n_proj_blocks": 1,
        # }
        # brain_decoder = BrainDecoder(**brain_decoder_params).to(device)
        # brain_decoder.load_state_dict(checkpoint["model_state_dict"])
        brain_decoder = BrainDecoder(**checkpoint["brain_decoder_params"]).to(device)
        brain_decoder.load_state_dict(checkpoint["brain_decoder_state_dict"])
        brain_decoder.eval()

        with open(Path(decoder_path).parent / "scale.pkl", "rb") as f:
            scale = pickle.load(f)
    else:
        with open(decoder_path, "rb") as f:
            brain_decoder = pickle.load(f)

    # Load brain features
    brain_features = []
    latent_features = []
    for dataset_id in dataset_ids:
        print(dataset_id, dataset_path, subject, latent_type, cache)
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
        brain_features.append(datamodule.train_data.betas)
        latent_features.append(datamodule.train_data.labels[latent_type])

    brain_features = np.concatenate(brain_features)
    latent_features = np.concatenate(latent_features)

    # Sub-sample brain features
    if (
        selected_indices_left_path is not None
        and selected_indices_right_path is not None
    ):
        with open(selected_indices_left_path, "rb") as f:
            selected_indices_left = np.load(f)

        with open(selected_indices_right_path, "rb") as f:
            selected_indices_right = np.load(f)

        n_vertices_left = get_n_vertices_left(subject)
        brain_features_left = brain_features[:, selected_indices_left]
        brain_features_right = brain_features[
            :, n_vertices_left + selected_indices_right
        ]
    else:
        brain_features_left = brain_features[:, :10242]
        brain_features_right = brain_features[:, 10242:]

    # Align brain features
    if left_mapping_path is not None and right_mapping_path is not None:
        left_mapping = load_mapping(left_mapping_path)
        right_mapping = load_mapping(right_mapping_path)

        if not invert_mapping:
            brain_features = np.concatenate(
                [
                    left_mapping.transform(brain_features_left),
                    right_mapping.transform(brain_features_right),
                ],
                axis=1,
            )
        else:
            brain_features = np.concatenate(
                [
                    left_mapping.inverse_transform(brain_features_left),
                    right_mapping.inverse_transform(brain_features_right),
                ],
                axis=1,
            )

    # Take opposite of brain features when subject is macaque
    # because they were scanned using MION instead of BOLD
    if subject_is_macaque:
        brain_features = -1 * brain_features

    # Predict latents
    if decoder_is_contrastive:
        dataloader = DataLoader(
            torch.from_numpy(brain_features),
            batch_size=128,
            shuffle=False,
            drop_last=False,
        )

        predictions = []
        predictions_reconstruction = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for brain_features_batch in dataloader:
                    brain_features_batch = brain_features_batch.to(
                        device, non_blocking=True
                    )
                    (
                        _,
                        predictions_contrastive_batch,
                        predictions_reconstruction_batch,
                    ) = brain_decoder(brain_features_batch)
                    predictions.append(predictions_contrastive_batch.cpu().numpy())
                    predictions_reconstruction.append(
                        predictions_reconstruction_batch.cpu().numpy()
                    )
        predictions = np.concatenate(predictions)
        predictions_reconstruction = np.concatenate(predictions_reconstruction)
    else:
        predictions = brain_decoder.predict(
            SimpleImputer().fit_transform(
                StandardScaler().fit_transform(brain_features)
            )
        )

    # Generate retrieval set
    generator = torch.Generator()
    generator.manual_seed(0)
    n_retrieval_set = 500
    retrieval_set_indices = torch.randperm(
        brain_features.shape[0], generator=generator
    )[: (min(n_retrieval_set, brain_features.shape[0]) - 1)]

    ground_truth = latent_features
    negatives = latent_features[retrieval_set_indices]

    # Evaluate
    retrieval_metrics = compute_retrieval_metrics(
        predictions=torch.from_numpy(predictions).to(torch.float32),
        ground_truth=torch.from_numpy(ground_truth).to(torch.float32),
        negatives=torch.from_numpy(negatives).to(torch.float32),
        return_scores=True,
    )

    retrieval_metrics["retrieval_set_indices"] = retrieval_set_indices.cpu().numpy()

    # Generate captions for each frame
    if should_generate_captions:
        if decoder_is_contrastive:
            prediction_scale = StandardScaler().fit(predictions_reconstruction)
            captions = generate_captions(
                scale.inverse_transform(
                    prediction_scale.transform(predictions_reconstruction)
                )
            )
        else:
            captions = generate_captions(predictions)

        with open(output_path / f"{output_name}_captions.pkl", "wb") as f:
            pickle.dump(captions, f)
    else:
        pass

    # Store results
    with open(output_path / f"{output_name}_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    if decoder_is_contrastive:
        with open(
            output_path / f"{output_name}_predictions_reconstruction.pkl", "wb"
        ) as f:
            pickle.dump(predictions_reconstruction, f)

    with open(output_path / f"{output_name}_scores.pkl", "wb") as f:
        pickle.dump(retrieval_metrics["scores"], f)

    del retrieval_metrics["scores"]

    with open(output_path / f"{output_name}_metrics.pkl", "wb") as f:
        pickle.dump(retrieval_metrics, f)
