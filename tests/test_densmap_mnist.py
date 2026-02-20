"""Test DensMAP implementation on MNIST (using sklearn digits as a fast proxy)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from sklearn.datasets import load_digits
from scipy.stats import pearsonr

from src.numap.umap_pytorch.main import PUMAP
from src.numap.umap_pytorch.modules import get_umap_graph
from src.numap.umap_pytorch.densmap import compute_R_orig, compute_R_emb
from src.numap.utils import get_spectral_embedding
from umap.umap_ import find_ab_params


def compute_density_correlation(X, embedding, n_neighbors=10, metric='euclidean'):
    """Compute Pearson correlation between R_orig and R_emb for evaluation."""
    graph = get_umap_graph(X, n_neighbors=n_neighbors, metric=metric)
    R_orig = compute_R_orig(graph, X, metric=metric)

    graph_coo = graph.tocoo()
    graph_coo.sum_duplicates()
    head = torch.as_tensor(graph_coo.row.copy(), dtype=torch.long)
    tail = torch.as_tensor(graph_coo.col.copy(), dtype=torch.long)
    n_vertices = graph_coo.shape[0]

    a, b = find_ab_params(1.0, 0.1)
    emb_tensor = torch.as_tensor(embedding, dtype=torch.float32)
    R_emb, valid_mask = compute_R_emb(emb_tensor, head, tail, a, b, n_vertices)

    corr, _ = pearsonr(R_orig[valid_mask].numpy(), R_emb[valid_mask].detach().numpy())
    return corr


def main():
    # Use sklearn digits: 1797 samples, 64 features (8x8 images, 0-9 digits)
    # This is a fast MNIST-like proxy for testing
    print("Loading digits dataset (MNIST-like, 1797 samples)...")
    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32)
    y = digits.target
    print(f"  X shape: {X.shape}, classes: {np.unique(y)}")

    n_neighbors = 10
    n_components = 2
    epochs = 20
    metric = 'euclidean'

    # Compute spectral embedding (required by NUMAP/PUMAP pipeline)
    print("\nComputing spectral embedding...")
    S = get_spectral_embedding(X, n_components=2, n_neighbors=10)

    # ---- Run 1: PUMAP without densmap (baseline) ----
    print("\n" + "=" * 60)
    print("Run 1: PUMAP WITHOUT densmap (baseline)")
    print("=" * 60)
    pumap_base = PUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        lr=1e-3,
        epochs=epochs,
        batch_size=64,
        learn_from_se=True,
        use_residual_connections=True,
        densmap=False,
    )
    pumap_base.fit(X, S)

    SX = torch.cat([S, X], dim=1)
    emb_base = pumap_base.transform(SX)

    corr_base = compute_density_correlation(X, emb_base, n_neighbors=n_neighbors, metric=metric)
    print(f"\n  Density correlation (R_orig vs R_emb): {corr_base:.4f}")

    # ---- Run 2: PUMAP with densmap ----
    print("\n" + "=" * 60)
    print("Run 2: PUMAP WITH densmap")
    print("=" * 60)
    pumap_dens = PUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        lr=1e-3,
        epochs=epochs,
        batch_size=64,
        learn_from_se=True,
        use_residual_connections=True,
        densmap=True,
        dens_lambda=2.0,
        dens_frac=0.3,
    )
    pumap_dens.fit(X, S)

    emb_dens = pumap_dens.transform(SX)

    corr_dens = compute_density_correlation(X, emb_dens, n_neighbors=n_neighbors, metric=metric)
    print(f"\n  Density correlation (R_orig vs R_emb): {corr_dens:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline (no densmap) density correlation: {corr_base:.4f}")
    print(f"  DensMAP density correlation:               {corr_dens:.4f}")
    print(f"  Improvement:                               {corr_dens - corr_base:+.4f}")
    if corr_dens > corr_base:
        print("  -> DensMAP improved density preservation!")
    else:
        print("  -> DensMAP did not improve (may need more epochs or tuning)")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
