"""Small helper utilities."""
import torch.nn.functional as F


def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)


def pair_accuracy(sim_pos, sim_neg):
    """Fraction of (positive, negative) pairs where sim_pos > sim_neg."""
    return (sim_pos.unsqueeze(1) > sim_neg.unsqueeze(0)).float().mean().item()
