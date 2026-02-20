import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from .data import UMAPDataset
from .modules import get_umap_graph, umap_loss
from .model import default_encoder, default_decoder, encoder_ft
from .densmap import compute_R_orig, compute_R_emb, DensityCorrelationLoss

from umap.umap_ import find_ab_params
import dill

""" Model """


class Model(pl.LightningModule):
    """PyTorch Lightning module for parametric UMAP training.

    Supports optional DensMAP density-preserving regularization. When
    ``densmap=True``, a density correlation loss is added to the UMAP
    cross-entropy loss during the last ``dens_frac`` fraction of training
    epochs. The density loss encourages the embedding to preserve local
    density structure from the original space.

    Args:
        lr: Learning rate for AdamW optimizer.
        encoder: Encoder network mapping input data to low-dimensional
            embeddings.
        decoder: Optional decoder for reconstruction loss.
        beta: Weight for reconstruction loss (if decoder is provided).
        min_dist: Minimum distance parameter for UMAP kernel.
        reconstruction_loss: Loss function for decoder reconstruction.
        match_nonparametric_umap: If True, match a pre-computed UMAP
            embedding using MSE instead of the standard UMAP loss.
        negative_sample_rate: Number of negative samples per positive edge.
        densmap: Enable DensMAP density-preserving regularization.
        dens_lambda: Weight of the density loss. Higher values produce
            stronger density preservation. Default: 2.0.
        dens_frac: Fraction of total epochs during which density loss is
            active (counted from the end). Default: 0.3 (last 30%).
        dens_var_shift: Numerical stability constant for Pearson correlation
            variance estimate. Default: 0.1.
        dens_subsample_size: Number of vertices to subsample per batch for
            density loss computation. Default: 10000.
        R_orig: Pre-computed local radii in original space (from
            ``compute_R_orig``). Required if ``densmap=True``.
        graph_head: Source vertex indices of the UMAP graph edges (numpy
            array). Required if ``densmap=True``.
        graph_tail: Target vertex indices of the UMAP graph edges (numpy
            array). Required if ``densmap=True``.
        n_vertices: Total number of vertices in the dataset. Required if
            ``densmap=True``.
        all_data: Full training data tensor (used for forward pass during
            density loss computation). Required if ``densmap=True``.
    """

    def __init__(
            self,
            lr: float,
            encoder: nn.Module,
            decoder=None,
            beta=1.0,
            min_dist=0.1,
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            match_nonparametric_umap=False,
            negative_sample_rate=5,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            dens_subsample_size=10000,
            R_orig=None,
            graph_head=None,
            graph_tail=None,
            n_vertices=None,
            all_data=None,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.negative_sample_rate = negative_sample_rate

        # densmap: stores graph edges and full dataset as buffers so they
        # automatically move to the correct device with the model
        self.densmap = densmap
        self.dens_frac = dens_frac
        self.dens_subsample_size = dens_subsample_size
        if densmap:
            self.dens_loss_fn = DensityCorrelationLoss(R_orig, dens_lambda, dens_var_shift)
            self.register_buffer('_graph_head_t', torch.as_tensor(graph_head, dtype=torch.long))
            self.register_buffer('_graph_tail_t', torch.as_tensor(graph_tail, dtype=torch.long))
            self.register_buffer('_all_data', all_data)
            self._n_vertices = n_vertices

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _should_apply_densmap(self):
        """Check if DensMAP loss should be active based on epoch progress.

        Returns True only during the last ``dens_frac`` fraction of total
        training epochs (e.g., the last 30% when ``dens_frac=0.3``).
        """
        if not self.densmap:
            return False
        progress = (self.current_epoch + 1) / self.trainer.max_epochs
        return progress > (1.0 - self.dens_frac)

    def _compute_density_loss(self):
        """Compute density correlation loss with vertex subsampling.

        For efficiency on large datasets, this method:
        1. Randomly subsamples ``dens_subsample_size`` vertices.
        2. Forwards the subsampled data through the encoder.
        3. Filters graph edges to only those within the subsample.
        4. Computes R_emb on the subsample using scatter operations.
        5. Returns -Pearson(R_orig[subsample], R_emb[subsample]).

        This provides a stochastic estimate of the full density correlation
        that is unbiased over training steps.

        Returns:
            Scalar loss tensor (0.0 if too few edges in subsample).
        """
        N = self._n_vertices
        sub_size = min(self.dens_subsample_size, N)

        # Random subsample of vertices
        sub_idx = torch.randperm(N, device=self.device)[:sub_size]

        # Forward subsampled data through encoder
        sub_data = self._all_data[sub_idx]
        sub_emb = self.encoder(sub_data)

        # Filter edges: keep only edges where BOTH endpoints are in subsample
        in_sub = torch.zeros(N, dtype=torch.bool, device=self.device)
        in_sub[sub_idx] = True
        edge_mask = in_sub[self._graph_head_t] & in_sub[self._graph_tail_t]

        # Remap edge indices to subsample indices
        remap = torch.full((N,), -1, dtype=torch.long, device=self.device)
        remap[sub_idx] = torch.arange(sub_size, device=self.device)
        sub_head = remap[self._graph_head_t[edge_mask]]
        sub_tail = remap[self._graph_tail_t[edge_mask]]

        # Need at least a few edges to compute meaningful correlation
        if sub_head.numel() < 10:
            return torch.tensor(0.0, device=self.device)

        # Compute R_emb on subsample
        R_emb_sub, valid_mask = compute_R_emb(sub_emb, sub_head, sub_tail, self._a, self._b, sub_size)
        R_orig_sub = self.dens_loss_fn.R_orig[sub_idx]

        # Only use vertices that have at least one edge in the subsample
        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=self.device)

        return self.dens_loss_fn.forward_with_subset(R_emb_sub[valid_mask], R_orig_sub[valid_mask])

    def training_step(self, batch, batch_idx):
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0],
                                     negative_sample_rate=self.negative_sample_rate)
            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self._should_apply_densmap():
                density_loss = self._compute_density_loss()
                # Scale density loss so its gradient magnitude is comparable
                # to the UMAP batch loss (density is computed over a much
                # larger subsample than the UMAP mini-batch)
                batch_size = edges_to_exp.shape[0]
                sub_size = min(self.dens_subsample_size, self._n_vertices)
                scale = batch_size / sub_size
                density_loss = density_loss * scale
                self.log("density_loss", density_loss, prog_bar=True)
                total_loss = encoder_loss + density_loss
            else:
                total_loss = encoder_loss

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return total_loss + self.beta * recon_loss
            else:
                return total_loss

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
            self,
            dataset,
            batch_size,
            num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=0,  # self.num_workers
            shuffle=True,
        )


class PUMAP():
    """Parametric UMAP with optional DensMAP density preservation.

    Wraps the Model class and handles graph construction, encoder creation,
    and training. When ``densmap=True``, the UMAP graph is constructed first,
    R_orig (local radii in original space) is pre-computed, and both are
    passed to the Model for density-regularized training.

    DensMAP parameters:
        densmap (bool): Enable density-preserving regularization. Default: False.
        dens_lambda (float): Density loss weight. Default: 2.0.
        dens_frac (float): Fraction of epochs (from the end) where density
            loss is active. Default: 0.3.
        dens_var_shift (float): Numerical stability for variance. Default: 0.1.
        dens_subsample_size (int): Vertices to subsample per step. Default: 10000.
    """

    def __init__(
            self,
            encoder=None,
            decoder=None,
            n_neighbors=10,
            min_dist=0.1,
            metric="euclidean",
            n_components=2,
            beta=1.0,
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            random_state=None,
            lr=1e-3,
            epochs=10,
            batch_size=64,
            num_workers=1,
            num_gpus=1,
            match_nonparametric_umap=False,
            use_residual_connections=False,
            learn_from_se=True,
            negative_sample_rate=5,
            use_concat=False,
            use_alpha=False,
            alpha=0.0,
            init_method='identity',
            model='numap',
            grease=None,
            frozen_layers=2,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            dens_subsample_size=10000,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        self.use_residual_connections = use_residual_connections
        self.learn_from_se = learn_from_se
        self.negative_sample_rate = negative_sample_rate
        self.use_concat = use_concat
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.init_method = init_method
        self.model = model
        self.grease = grease
        self.frozen_layers = frozen_layers
        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.dens_subsample_size = dens_subsample_size

    def fit(self, X, S):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model == 'numap':
            SX = torch.cat([S, X], dim=1)
            if self.learn_from_se:
                input_dims = S.shape[1:]
            elif self.use_concat:
                input_dims = S.shape[1] + X.shape[1]
            else:
                input_dims = X.shape[1:]

            trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                max_epochs=self.epochs
            )
            # encoder = default_encoder(X.shape[1:], self.n_components) if self.encoder is None else self.encoder
            encoder = default_encoder(input_dims, self.n_components, self.use_residual_connections, self.learn_from_se,
                                      self.use_concat, self.use_alpha, self.alpha,
                                      S, self.init_method, device=device) if self.encoder is None else self.encoder

            if self.decoder is None or isinstance(self.decoder, nn.Module):
                decoder = self.decoder
            elif self.decoder == True:
                # decoder = default_decoder(X.shape[1:], self.n_components)
                decoder = default_decoder(S.shape[1:], self.n_components)

            # Build graph BEFORE model so we can compute R_orig for densmap
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)

            # Prepare densmap data if enabled
            densmap_kwargs = {}
            if self.densmap:
                R_orig = compute_R_orig(graph, X, metric=self.metric)
                graph_coo = graph.tocoo()
                graph_coo.sum_duplicates()
                densmap_kwargs = dict(
                    densmap=True,
                    dens_lambda=self.dens_lambda,
                    dens_frac=self.dens_frac,
                    dens_var_shift=self.dens_var_shift,
                    dens_subsample_size=self.dens_subsample_size,
                    R_orig=R_orig,
                    graph_head=graph_coo.row.copy(),
                    graph_tail=graph_coo.col.copy(),
                    n_vertices=graph_coo.shape[0],
                    all_data=torch.as_tensor(SX, dtype=torch.float32),
                )

            self.model = Model(self.lr, encoder, decoder, beta=self.beta, min_dist=self.min_dist,
                               reconstruction_loss=self.reconstruction_loss,
                               negative_sample_rate=self.negative_sample_rate,
                               **densmap_kwargs)

            trainer.fit(
                model=self.model,
                # datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
                datamodule=Datamodule(UMAPDataset(SX, graph), self.batch_size, self.num_workers)
            )
        elif self.model == 'numap_ft':
            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
            encoder = encoder_ft(X.shape[1:], self.n_components, self.grease._spectralnet.spec_net, self.grease.ortho_matrix, self.frozen_layers)

            self.model = Model(self.lr, encoder, None, beta=self.beta, min_dist=self.min_dist,
                               reconstruction_loss=self.reconstruction_loss,
                               negative_sample_rate=self.negative_sample_rate)
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
            )

    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")


def load_pumap(path):
    print("Loading PUMAP object from pickled file.")
    with open(path, 'rb') as inp: return dill.load(inp)


if __name__ == "__main__":
    pass
