import torch
import numpy as np


def nai(datasets, zsh, ft, merged):
    # caluclate normalized accuracy improvement (from Marczak et al.)
    return {ds: (merged[ds] - zsh[ds]) / (ft[ds] - zsh[ds]) for ds in datasets}


def calc_rank(S, norm_thresh=0.95):
    # Rank based on approximation error (Eq. 6) in the paper
    rank = np.argmax(np.sqrt(np.cumsum(S.pow(2) / S.pow(2).sum())) > norm_thresh)
    return rank


def alignment_ratio(S, S_proj):
    # Subspace alignment ratio based on norms of projected task matrix vs norm of the original one (Eq. 5) in the paper
    return np.linalg.norm(S_proj, ord=2) / np.linalg.norm(S, ord=2)


@torch.no_grad()
def sar():
    pass
