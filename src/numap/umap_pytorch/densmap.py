"""DensMAP: Density-preserving regularization for parametric UMAP.

Adds a density preservation term to the UMAP loss that encourages the embedding
to preserve local density structure from the original high-dimensional space.
The key idea is to compute "local radii" for each vertex in both the original
and embedding spaces, then penalize low Pearson correlation between them.

Mathematical formulation:
    R_orig[i] = log(eps + sum_j(mu_ij * d^2(x_i, x_j)) / sum_j(mu_ij))
    R_emb[i]  = log(eps + sum_j(phi_ij * d^2(y_i, y_j)) / sum_j(phi_ij))
    L_density  = -lambda * PearsonCorrelation(R_orig, R_emb)

where mu_ij are the fuzzy membership weights from the UMAP graph, and
phi_ij = 1/(1 + a*d^(2b)) is the UMAP kernel in embedding space.

Reference:
    Narayan, A., Berger, B., & Cho, H. (2021). Density-Preserving Data
    Visualization Unveils Dynamic Patterns of Single-Cell Transcriptomic
    Variability. Nature Biotechnology.
"""

import numpy as np
import torch
from torch import nn


def compute_R_orig(graph, data, metric='euclidean'):
    """Compute normalized local radii from the UMAP graph in original space.

    For each vertex i, the local radius captures its local density as:
        R[i] = log(eps + sum_j(mu_ij * d^2(x_i, x_j)) / sum_j(mu_ij))
    The result is standardized to mean=0, std=1.

    This function runs once before training (numpy, not differentiable).

    Args:
        graph: Sparse UMAP graph (scipy sparse matrix) with fuzzy membership
            weights. Typically produced by ``get_umap_graph()``.
        data: Original high-dimensional data, shape (n_samples, n_features).
            Can be a numpy array or torch Tensor.
        metric: Distance metric used to build the graph. Supports 'euclidean',
            'cosine', or any metric accepted by sklearn's pairwise_distances.

    Returns:
        torch.Tensor: Standardized local radii R_orig, shape (n_vertices,).
    """
    graph_coo = graph.tocoo()
    graph_coo.sum_duplicates()
    head, tail, mu = graph_coo.row, graph_coo.col, graph_coo.data
    n_vertices = graph_coo.shape[0]

    data_np = np.asarray(data)
    head_data = data_np[head]
    tail_data = data_np[tail]

    if metric == 'euclidean':
        dists_sq = np.sum((head_data - tail_data) ** 2, axis=1)
    elif metric == 'cosine':
        cos_sim = np.sum(head_data * tail_data, axis=1) / (
            np.linalg.norm(head_data, axis=1) * np.linalg.norm(tail_data, axis=1) + 1e-8)
        dists_sq = (1 - cos_sim) ** 2
    else:
        from sklearn.metrics import pairwise_distances
        dists_sq = np.array([
            pairwise_distances(h.reshape(1, -1), t.reshape(1, -1), metric=metric)[0, 0] ** 2
            for h, t in zip(head_data, tail_data)
        ])

    ro = np.zeros(n_vertices)
    mu_sum = np.zeros(n_vertices)
    np.add.at(ro, head, mu * dists_sq)
    np.add.at(ro, tail, mu * dists_sq)
    np.add.at(mu_sum, head, mu)
    np.add.at(mu_sum, tail, mu)

    ro = np.log(1e-8 + ro / mu_sum)
    std = ro.std()
    if std < 1e-10:
        std = 1.0
    R_orig = (ro - ro.mean()) / std
    return torch.tensor(R_orig, dtype=torch.float32)


def compute_R_emb(embeddings, head, tail, a, b, n_vertices):
    """Compute local radii in the embedding space (fully differentiable).

    Uses the UMAP kernel phi_ij = 1/(1 + a*d^(2b)) to weight distances:
        R_emb[i] = log(eps + sum_j(phi_ij * d^2(y_i, y_j)) / sum_j(phi_ij))

    Uses ``index_add_`` scatter operations for efficient GPU computation.

    Args:
        embeddings: Embedding coordinates, shape (n_vertices, n_components).
            Must have ``requires_grad=True`` for backpropagation.
        head: Source vertex indices for each edge, shape (n_edges,).
            Long tensor on the same device as embeddings.
        tail: Target vertex indices for each edge, shape (n_edges,).
            Long tensor on the same device as embeddings.
        a: UMAP kernel parameter (from ``find_ab_params``).
        b: UMAP kernel parameter (from ``find_ab_params``).
        n_vertices: Total number of vertices.

    Returns:
        tuple: (R_emb, valid_mask) where:
            - R_emb: Local radii tensor, shape (n_vertices,).
            - valid_mask: Boolean tensor, shape (n_vertices,). True for
              vertices that have at least one edge (vertices with no edges
              produce NaN and should be excluded from the loss).
    """
    dists_sq = ((embeddings[head] - embeddings[tail]) ** 2).sum(dim=1)
    phi = 1.0 / (1.0 + a * dists_sq.pow(b))
    weighted_dists = phi * dists_sq

    re_sum = torch.zeros(n_vertices, device=embeddings.device)
    phi_sum = torch.zeros(n_vertices, device=embeddings.device)
    re_sum.index_add_(0, head, weighted_dists)
    re_sum.index_add_(0, tail, weighted_dists)
    phi_sum.index_add_(0, head, phi)
    phi_sum.index_add_(0, tail, phi)

    valid_mask = phi_sum > 0
    # Add epsilon to phi_sum to avoid division by zero
    R_emb = torch.log(1e-8 + re_sum / (phi_sum + 1e-10))
    return R_emb, valid_mask


class DensityCorrelationLoss(nn.Module):
    """Density preservation loss: -lambda * Pearson(R_orig, R_emb).

    Minimizing this loss maximizes the Pearson correlation between local
    radii in the original and embedding spaces, encouraging the embedding
    to preserve local density structure.

    The variance shift ``dens_var_shift`` is added to the variance estimates
    for numerical stability (prevents division by zero when all radii are
    equal).

    Args:
        R_orig: Pre-computed local radii in original space, shape (n_vertices,).
            Stored as a buffer (moves with the model to GPU automatically).
        dens_lambda: Weight of the density regularization term. Higher values
            produce stronger density preservation at the cost of UMAP
            structure. Default: 2.0.
        dens_var_shift: Small constant added to variance for numerical
            stability. Default: 0.1.
    """

    def __init__(self, R_orig, dens_lambda=2.0, dens_var_shift=0.1):
        super().__init__()
        self.register_buffer('R_orig', R_orig)
        self.dens_lambda = dens_lambda
        self.dens_var_shift = dens_var_shift

    def forward(self, R_emb):
        """Compute loss using the full R_orig stored in this module.

        Args:
            R_emb: Local radii in embedding space, shape (n_vertices,).

        Returns:
            Scalar loss tensor.
        """
        return self._pearson_loss(self.R_orig, R_emb)

    def forward_with_subset(self, R_emb, R_orig_sub):
        """Compute loss using a subset of vertices (for stochastic estimation).

        Args:
            R_emb: Embedding radii for the subset, shape (subset_size,).
            R_orig_sub: Original radii for the subset, shape (subset_size,).

        Returns:
            Scalar loss tensor.
        """
        return self._pearson_loss(R_orig_sub, R_emb)

    def _pearson_loss(self, R_orig, R_emb):
        """Compute -lambda * Pearson correlation between R_orig and R_emb."""
        ro = R_orig - R_orig.mean()
        re = R_emb - R_emb.mean()
        cov = (ro * re).mean()
        std_o = torch.sqrt(ro.var() + self.dens_var_shift)
        std_e = torch.sqrt(re.var() + self.dens_var_shift)
        return -self.dens_lambda * cov / (std_o * std_e)
