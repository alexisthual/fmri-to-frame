import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import norm


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

    r = np.sum(
        (xm / normxm[:, np.newaxis]) * (ym / normym[:, np.newaxis]), axis=1
    )

    return r


def compute_retrieval_metrics(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    negatives: torch.Tensor,
    should_plot: bool = False,
    return_scores: bool = False,
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
    inv_norms_negatives = 1 / (
        1e-16 + negatives.to(torch.float32).norm(dim=1, p=2)
    )
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
    for k in [1, 5, 10]:
        # It can be that the current batch is smaller than k,
        # in which case topk would raise an error.
        min_k = min(k, all_scores.shape[1])
        accuracy = (ranks < min_k).sum() / ranks.shape[0]
        outputs.update({f"top-{k}_accuracy": accuracy.item()})

    if return_scores:
        outputs.update({"scores": all_scores})

    return outputs
