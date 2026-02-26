# NUMAP

<p align="center">

[//]: # (This is my PyTorch implementation of NUMAP, a new and generalizable UMAP implementation, from the paper ["Generalizable Spectral Embedding with Applications to UMAP"]&#40;&#41;.<br>)
This is the official PyTorch implementation of NUMAP, a new and generalizable UMAP implementation.

See our [GitHub repository](https://github.com/shaham-lab/NUMAP) for more information and the latest updates.

[//]: # (## Installation)

[//]: # (You can install the latest package version via)

[//]: # (```bash)
[//]: # (pip install spectralnet)
[//]: # (```)

NUMAP can be used to visualize many types of data in a low-dimensional space, while enabling a simple out-of-sample extension.
One application of NUMAP is to **visualize time-series data**, and help understand the process in a given system.
For example, the following figure shows the transition of a set of points from one state to another, using NUMAP.
In a biological point of view, this can be viewed as a simplified simulation of the cellular differentiation process.

[//]: # (github)
<img src="figures\NUMAP_timesteps_transition_1color.png">

[//]: # (pypi)
[//]: # (<img src="https://github.com/shaham-lab/NUMAP/raw/main/figures/NUMAP_timesteps_transition_1color.png">)

The package is based on UMAP and [**GrEASE (Generalizable and Efficient Approximate Spectral Embedding)**](https://github.com/shaham-lab/GrEASE).
It is easy to use and can be used with any PyTorch dataset, on both CPU and GPU.
The package also includes a test dataset and a test script to run the model on the 2 Circles dataset.

The incorporation of GrEASE enables preservation of both **local and global structures** of the data, as UMAP,
with the new capability of out-of-sample extension.

[//]: # (github)
<img src="figures\intro_fig_idsai_colored.png">
    
[//]: # (pypi)
[//]: # (<img src="https://github.com/shaham-lab/NUMAP/raw/main/figures/intro_fig_idsai_colored.png">)

## Installation

Install the latest stable release from PyPI:

```bash
pip install numap
```

To install the latest development version (with DensMAP support) directly from GitHub:

```bash
pip install git+https://github.com/chansigit/NUMAP.git
```

## Usage

The basic functionality is quite intuitive and easy to use, e.g.,

```python
from numap import NUMAP

numap = NUMAP(n_components=2)  # n_components is the number of dimensions in the low-dimensional representation
numap.fit(X)  # X is the dataset and it should be a torch.Tensor
X_reduced = numap.transform(X)  # Get the low-dimensional representation of the dataset
Y_reduced = numap.transform(Y)  # Get the low-dimensional representation of a test dataset
```

You can read the code docs for more information and functionalities.<br>

## DensMAP: Density-Preserving Visualization

NUMAP supports **DensMAP**, a density-preserving extension of UMAP that encourages the
low-dimensional embedding to preserve the local density structure of the original
high-dimensional data. This is useful when the relative densities of clusters carry
meaningful information (e.g., cell population sizes in single-cell RNA-seq).

### Quick start

```python
from numap import NUMAP

numap = NUMAP(
    n_neighbors=10,
    epochs=50,
    lr=1e-3,
    densmap=True,           # enable density preservation
    dens_lambda=2.0,        # density regularization strength
    dens_frac=0.3,          # activate in last 30% of epochs
)
numap.fit(X)                # X is a torch.Tensor
embedding = numap.transform(X)
```

### DensMAP Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `densmap` | `False` | Enable density-preserving regularization. |
| `dens_lambda` | `2.0` | Weight of the density loss. Higher values give stronger density preservation at the cost of cluster separation. Recommended range: 0.1–2.0. |
| `dens_frac` | `0.3` | Fraction of total epochs (from the end) during which the density loss is active. For example, `0.3` means the last 30% of epochs. |
| `dens_var_shift` | `0.1` | Small constant for numerical stability in the Pearson correlation. |
| `dens_subsample_size` | `10000` | Number of vertices randomly subsampled per training step for density loss. Larger = more accurate but more memory. |

### How it works

DensMAP adds a regularization term to the UMAP loss:

**L = L_UMAP + L_density**

where L_density = −λ · Pearson(R_orig, R_emb).

- **R_orig** captures the local density around each point in the original space, computed from the UMAP fuzzy graph (pre-computed once before training).
- **R_emb** captures the local density in the embedding space, recomputed during training using a differentiable UMAP kernel.
- Minimizing −Pearson(R_orig, R_emb) pushes the embedding to preserve relative densities.

The density loss is only activated during the last `dens_frac` fraction of training, allowing the UMAP structure to form first before density refinement.

### Tuning tips

- Start with `dens_lambda=0.5` for a good balance between cluster separation and density preservation.
- Increase `dens_lambda` (up to 2.0) if density preservation is more important than cluster separation.
- Decrease `dens_lambda` (e.g., 0.1) if clusters are merging too much.
- Use more `epochs` (50+) when DensMAP is enabled, since the density phase needs enough steps to converge.

## Running examples

In order to run the model on the 2 Circles dataset, you can either run the file, or using the command-line command:<br>
`python tests/run_numap.py`<br>
This will run NUMAP and UMAP on the 2 Circles dataset and plot the results.




[//]: # (## Citation)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (@inproceedings{shaham2018,)

[//]: # (author = {Uri Shaham and Kelly Stanton and Henri Li and Boaz Nadler and Ronen Basri and Yuval Kluger},)

[//]: # (title = {SpectralNet: Spectral Clustering Using Deep Neural Networks},)

[//]: # (booktitle = {Proc. ICLR 2018},)

[//]: # (year = {2018})

[//]: # (})

[//]: # ()
[//]: # (```)

