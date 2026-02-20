"""Run DensMAP on MNIST with multiple lambda values, save comparison to PDF."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from torchvision import datasets, transforms

from src.numap.umap_pytorch.main import PUMAP
from src.numap.umap_pytorch.modules import get_umap_graph
from src.numap.umap_pytorch.densmap import compute_R_orig, compute_R_emb
from src.numap.utils import get_spectral_embedding
from umap.umap_ import find_ab_params

torch.set_float32_matmul_precision('medium')


def compute_density_correlation(X, embedding, n_neighbors=10, metric='euclidean'):
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


def load_mnist(subset_size=10000):
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="../data", train=True, download=True, transform=tensor_transform)
    x_train, y_train = zip(*train_set)
    x_train = torch.cat(x_train).view(len(x_train), -1)
    y_train = torch.tensor(y_train)
    idx = torch.randperm(len(x_train))[:subset_size]
    return x_train[idx], y_train[idx].numpy()


def run_pumap(X, S, densmap=False, dens_lambda=2.0, epochs=50):
    pumap = PUMAP(
        n_neighbors=10,
        n_components=2,
        metric='euclidean',
        lr=1e-3,
        epochs=epochs,
        batch_size=256,
        learn_from_se=False,
        use_residual_connections=True,
        densmap=densmap,
        dens_lambda=dens_lambda,
        dens_frac=0.3,
        dens_subsample_size=5000,
    )
    pumap.fit(X, S)
    return pumap


def main():
    subset_size = 10000
    epochs = 50

    print(f"Loading MNIST ({subset_size} samples)...")
    X, y = load_mnist(subset_size=subset_size)
    print(f"  X shape: {X.shape}")

    print("\nComputing spectral embedding...")
    S = get_spectral_embedding(X, n_components=2, n_neighbors=10)
    SX = torch.cat([S, X], dim=1)

    # Configs: (label, densmap, dens_lambda)
    configs = [
        ("Baseline", False, 0.0),
        ("DensMAP λ=0.1", True, 0.1),
        ("DensMAP λ=0.5", True, 0.5),
        ("DensMAP λ=2.0", True, 2.0),
    ]

    results = []
    for label, densmap, lam in configs:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        pumap = run_pumap(X, S, densmap=densmap, dens_lambda=lam, epochs=epochs)
        emb = pumap.transform(SX)
        corr = compute_density_correlation(X, emb)
        print(f"  Density correlation: {corr:.4f}")
        results.append((label, emb, corr))

    # ---- Save to PDF ----
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mnist_densmap.pdf'))

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for i, (label, emb, corr) in enumerate(results):
        sc = axes[i].scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=0.5, alpha=0.7)
        axes[i].set_title(label, fontsize=13)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].text(0.5, -0.05, f'ρ = {corr:.3f}',
                     transform=axes[i].transAxes, ha='center', fontsize=11)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.01)
    cbar.set_label('Digit', fontsize=11)
    cbar.set_ticks(range(10))
    fig.suptitle(f'MNIST ({subset_size} samples, {epochs} epochs)', fontsize=15, y=1.01)
    plt.tight_layout()

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPDF saved to: {pdf_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, _, corr in results:
        print(f"  {label:20s}  ρ = {corr:.4f}")


if __name__ == "__main__":
    main()
