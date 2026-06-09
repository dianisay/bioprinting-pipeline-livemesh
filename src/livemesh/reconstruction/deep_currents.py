"""
DeepCurrents: Neural implicit surface reconstruction with explicit boundary control.

Based on Palmer, Smirnov, Wang, Chern & Solomon (CVPR 2022):
"DeepCurrents: Learning Implicit Representations of Shapes with Boundaries"

The method represents a surface as a 2-current via the Hodge decomposition:

    omega = df_theta + alpha_Gamma

where f_theta is a neural network and alpha_Gamma is the Biot-Savart field
of the boundary curve Gamma. Training minimizes the mass norm (area) of the
current under a data-dependent Riemannian metric, producing an implicit surface
that (a) interpolates the prescribed boundary exactly and (b) fits the target
geometry from point cloud or mesh data.

For bioprinting: the wound boundary from segmentation becomes the explicit
boundary curve, and the depth-camera point cloud provides the target geometry.
The reconstructed surface supports geodesic toolpath generation with boundary-
aware nozzle trajectories.

Reference:
    @inproceedings{palmer2022deepcurrents,
        title  = {DeepCurrents: Learning Implicit Representations of
                  Shapes with Boundaries},
        author = {Palmer, David and Smirnov, Dmitriy and Wang, Stephanie
                  and Chern, Albert and Solomon, Justin},
        booktitle = {CVPR},
        year   = {2022},
        pages  = {18665--18675},
    }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepCurrentsConfig:
    """All hyperparameters for DeepCurrents training and meshing."""

    # Network
    rff_dim: int = 2048
    rff_sigma: float = 2.0
    hidden_dim: int = 256
    num_hidden_layers: int = 4

    # Training
    lr: float = 1e-3
    lr_decay: float = 0.6
    lr_decay_every: int = 2000
    n_iterations: int = 10_000
    n_samples_uniform: int = 4000
    n_samples_surface: int = 4000

    # Loss
    surface_loss_delta: float = 0.01
    surface_loss_eps_min: float = 0.001
    surface_loss_eps_max: float = 0.02
    boundary_weight_sigma: float = 0.1
    biot_savart_scale: float = 1e-3

    # Meshing
    marching_cubes_resolution: int = 128
    current_prune_threshold: float = 5e-3

    # Domain
    domain_min: float = -1.0
    domain_max: float = 1.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every: int = 500


# ---------------------------------------------------------------------------
# Random Fourier Features
# ---------------------------------------------------------------------------

class RandomFourierFeatures(nn.Module):
    """Project low-dimensional coordinates into a high-dimensional feature space.

    Uses the random Fourier feature (RFF) mapping from Tancik et al. (2020):
        gamma(x) = [sin(2*pi*B*x), cos(2*pi*B*x)]
    where B is sampled from N(0, sigma^2).

    This is essential for learning high-frequency surface details; without it
    the MLP cannot represent sharp features (see ablation in Palmer et al.).
    """

    def __init__(self, d_in: int = 3, d_out: int = 2048, sigma: float = 2.0):
        super().__init__()
        self.register_buffer(
            "B", torch.randn(d_out // 2, d_in) * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = (2.0 * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


# ---------------------------------------------------------------------------
# Neural network f_theta
# ---------------------------------------------------------------------------

class CurrentMLP(nn.Module):
    """MLP that maps RFF features to a scalar field f(x).

    The current is then j(x) = grad_x f(x) + alpha(x), where alpha is
    the Biot-Savart field. The gradient is computed via torch.autograd.grad.

    Architecture follows Palmer et al.: hidden_dim=256, 4 hidden layers,
    Softplus activations (NOT ReLU: the zero second derivative of ReLU
    blurs the learned current; see ablation in the paper).
    """

    def __init__(self, d_in: int = 2048, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.Softplus())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ---------------------------------------------------------------------------
# Biot-Savart field
# ---------------------------------------------------------------------------

def biot_savart_3d(
    x: torch.Tensor,
    boundary_verts: torch.Tensor,
    scale: float = 1e-3,
) -> torch.Tensor:
    """Compute the Biot-Savart vector field of a closed polygonal boundary.

    Given query points x and a closed boundary curve defined by ordered vertices,
    returns the vector field alpha^sharp(x) that satisfies d(alpha) = delta_Gamma
    (the boundary acts as a magnetic current source, and alpha is the induced field).

    For a polygonal curve with segments (v_i, v_{i+1}), the field at x is:

        alpha^sharp(x) = sum_i  (t_hat_i . (r_hat1_i - r_hat0_i)) * (t_hat_i x r0_i)
                                / |t_hat_i x r0_i|^2

    Parameters
    ----------
    x : (M, 3) query points
    boundary_verts : (L, V, 3) boundary loops (L loops, V vertices each)
    scale : normalization factor (default 1e-3 matches paper)

    Returns
    -------
    alpha : (M, 3) vector field
    """
    if boundary_verts.dim() == 2:
        boundary_verts = boundary_verts.unsqueeze(0)

    result = torch.zeros_like(x)

    for loop_verts in boundary_verts:
        v0 = loop_verts                                    # (V, 3)
        v1 = torch.roll(loop_verts, -1, dims=0)            # (V, 3)

        edges = v1 - v0                                     # (V, 3)
        tangents = F.normalize(edges, p=2, dim=-1)          # (V, 3)

        # r0_i = x - v0_i, r1_i = x - v1_i
        r0 = x.unsqueeze(1) - v0.unsqueeze(0)              # (M, V, 3)
        r1 = x.unsqueeze(1) - v1.unsqueeze(0)              # (M, V, 3)

        r0_hat = F.normalize(r0, p=2, dim=-1)              # (M, V, 3)
        r1_hat = F.normalize(r1, p=2, dim=-1)              # (M, V, 3)

        t = tangents.unsqueeze(0)                           # (1, V, 3)

        cross = torch.cross(t.expand_as(r0), r0, dim=-1)   # t x r0: (M, V, 3)
        perp_sq = cross.pow(2).sum(-1, keepdim=True).clamp(min=1e-10)  # (M, V, 1)

        dot_diff = (r1_hat * t).sum(-1, keepdim=True) - \
                   (r0_hat * t).sum(-1, keepdim=True)       # (M, V, 1)

        contrib = cross * dot_diff / perp_sq                # (M, V, 3)
        result = result + contrib.sum(dim=1)                # (M, 3)

    return result * scale


# ---------------------------------------------------------------------------
# Full DeepCurrents model
# ---------------------------------------------------------------------------

class DeepCurrentsModel(nn.Module):
    """Neural implicit surface with explicit boundary via the theory of currents.

    Combines:
    - Random Fourier Features for positional encoding
    - MLP f_theta for the scalar potential
    - Biot-Savart field alpha for boundary enforcement
    - Autograd for gradient computation (df)

    The current j(x) = grad f(x) + alpha(x) represents the surface normal
    direction; the surface is where |j| concentrates.
    """

    def __init__(self, config: DeepCurrentsConfig, boundary_verts: torch.Tensor):
        super().__init__()
        self.config = config
        self.rff = RandomFourierFeatures(
            d_in=3, d_out=config.rff_dim, sigma=config.rff_sigma
        )
        self.mlp = CurrentMLP(
            d_in=config.rff_dim, hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers
        )
        self.register_buffer("boundary_verts", boundary_verts)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the current at query points.

        Parameters
        ----------
        x : (N, 3) query points with requires_grad=True
        return_components : if True, return df and alpha separately

        Returns
        -------
        dict with keys: 'f', 'current', and optionally 'df', 'alpha'
        """
        x_clamped = x.clamp(self.config.domain_min, self.config.domain_max)
        features = self.rff(x_clamped)
        f = self.mlp(features)

        df = torch.autograd.grad(
            outputs=f, inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True, retain_graph=True,
        )[0]

        alpha = biot_savart_3d(
            x_clamped, self.boundary_verts, scale=self.config.biot_savart_scale
        )

        current = df + alpha

        out = {"f": f, "current": current}
        if return_components:
            out["df"] = df
            out["alpha"] = alpha
        return out

    def evaluate_f(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate only the scalar field f (no gradient computation)."""
        x_clamped = x.clamp(self.config.domain_min, self.config.domain_max)
        features = self.rff(x_clamped)
        return self.mlp(features)


# ---------------------------------------------------------------------------
# Metric and loss computation
# ---------------------------------------------------------------------------

def compute_anisotropic_metric(
    current: torch.Tensor,
    closest_normals: torch.Tensor,
    is_boundary: torch.Tensor,
) -> torch.Tensor:
    """Apply the data-dependent metric B_x to the current vector.

    B_x = (I - n*n^T) for interior points (penalizes deviation from target normal)
    B_x = I            for boundary-adjacent points (no penalty near boundary)

    The transformation: B_x * v = v - (n . v) * n  (for interior)
                        B_x * v = v                 (for boundary)

    Parameters
    ----------
    current : (N, 3) current vectors
    closest_normals : (N, 3) normals of closest faces on target mesh
    is_boundary : (N,) boolean mask for boundary-adjacent points
    """
    n_dot_v = (closest_normals * current).sum(dim=-1, keepdim=True)
    normal_component = closest_normals * n_dot_v

    transformed = current.clone()
    interior_mask = ~is_boundary
    transformed[interior_mask] = current[interior_mask] - normal_component[interior_mask]

    return transformed


def compute_boundary_weights(
    points: torch.Tensor,
    boundary_verts: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Gaussian boundary weighting: w(x) = exp(-dist^2 / 2*sigma^2).

    Points closer to the boundary get higher weight in the current loss,
    which sharpens the reconstruction near the prescribed boundary.

    Parameters
    ----------
    points : (N, 3)
    boundary_verts : (L, V, 3) boundary loop vertices
    sigma : Gaussian falloff parameter

    Returns
    -------
    weights : (N,)
    """
    if boundary_verts.dim() == 2:
        boundary_verts = boundary_verts.unsqueeze(0)

    min_dist_sq = torch.full((points.shape[0],), float("inf"), device=points.device)

    for loop_verts in boundary_verts:
        v0 = loop_verts                                     # (V, 3)
        v1 = torch.roll(loop_verts, -1, dims=0)             # (V, 3)
        edges = v1 - v0                                      # (V, 3)
        edge_len_sq = edges.pow(2).sum(-1).clamp(min=1e-12)  # (V,)

        # Project each point onto each edge segment
        diff = points.unsqueeze(1) - v0.unsqueeze(0)         # (N, V, 3)
        t = (diff * edges.unsqueeze(0)).sum(-1) / edge_len_sq.unsqueeze(0)
        t = t.clamp(0.0, 1.0)                                # (N, V)

        closest = v0.unsqueeze(0) + t.unsqueeze(-1) * edges.unsqueeze(0)  # (N, V, 3)
        dist_sq = (points.unsqueeze(1) - closest).pow(2).sum(-1)          # (N, V)
        min_per_loop = dist_sq.min(dim=1).values                          # (N,)
        min_dist_sq = torch.min(min_dist_sq, min_per_loop)

    return torch.exp(-min_dist_sq / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

@dataclass
class PreparedTarget:
    """Preprocessed target surface data for training."""

    vertices: torch.Tensor
    faces: torch.Tensor
    face_normals: torch.Tensor
    boundary_verts: torch.Tensor
    boundary_edges: torch.Tensor
    surface_points: torch.Tensor
    surface_normals: torch.Tensor
    normalization_center: NDArray[np.float64]
    normalization_scale: float


def prepare_target(
    mesh: trimesh.Trimesh,
    boundary_vertices: NDArray[np.float64] | None = None,
    n_surface_samples: int = 10_000,
    device: str = "cpu",
) -> PreparedTarget:
    """Normalize a target mesh to [-0.5, 0.5]^3 and extract boundary + surface samples.

    Parameters
    ----------
    mesh : target surface (open mesh with boundary, or closed mesh)
    boundary_vertices : (V, 3) ordered boundary vertices. If None, extracted
                       automatically from mesh boundary edges.
    n_surface_samples : number of surface points to pre-sample for L_surf
    device : torch device
    """
    verts = np.array(mesh.vertices, dtype=np.float64)

    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts_centered = verts - center
    scale = np.abs(verts_centered).max() * 2.2
    verts_norm = verts_centered / scale

    logger.info(
        f"Normalized mesh to [-0.5, 0.5]^3: center={center}, scale={scale:.4f}"
    )

    if boundary_vertices is None:
        boundary_vertices = _extract_boundary(mesh)
        if boundary_vertices is not None:
            boundary_vertices = (boundary_vertices - center) / scale
            logger.info(f"Auto-extracted boundary: {len(boundary_vertices)} vertices")
        else:
            raise ValueError(
                "Mesh has no boundary edges. DeepCurrents requires an open surface "
                "with at least one boundary loop. Use Poisson reconstruction for "
                "closed surfaces."
            )
    else:
        boundary_vertices = (boundary_vertices - center) / scale

    faces = np.array(mesh.faces, dtype=np.int64)
    mesh_norm = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=False)
    face_normals = np.array(mesh_norm.face_normals, dtype=np.float64)

    surf_pts, face_idx = trimesh.sample.sample_surface(mesh_norm, n_surface_samples)
    surf_normals = face_normals[face_idx]

    bdry_edges_list = []
    bdry_v = boundary_vertices
    for i in range(len(bdry_v)):
        j = (i + 1) % len(bdry_v)
        bdry_edges_list.append(np.stack([bdry_v[i], bdry_v[j]]))
    bdry_edges = np.array(bdry_edges_list)

    return PreparedTarget(
        vertices=torch.tensor(verts_norm, dtype=torch.float32, device=device),
        faces=torch.tensor(faces, dtype=torch.long, device=device),
        face_normals=torch.tensor(face_normals, dtype=torch.float32, device=device),
        boundary_verts=torch.tensor(boundary_vertices, dtype=torch.float32, device=device),
        boundary_edges=torch.tensor(bdry_edges, dtype=torch.float32, device=device),
        surface_points=torch.tensor(surf_pts, dtype=torch.float32, device=device),
        surface_normals=torch.tensor(surf_normals, dtype=torch.float32, device=device),
        normalization_center=center,
        normalization_scale=scale,
    )


def _extract_boundary(mesh: trimesh.Trimesh) -> NDArray[np.float64] | None:
    """Extract ordered boundary loop vertices from an open mesh."""
    try:
        edges = mesh.edges_unique
        edge_face_count = trimesh.grouping.group_rows(
            mesh.edges_sorted, require_count=2
        )
    except Exception:
        edge_face_count = None

    boundary_edge_mask = np.array(mesh.edges_unique_length) > 0
    face_adjacency = mesh.face_adjacency_edges

    boundary_edges = []
    edge_to_faces = {}
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_to_faces.setdefault(e, []).append(fi)

    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            boundary_edges.append(edge)

    if not boundary_edges:
        return None

    adj = {}
    for e in boundary_edges:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])

    start = boundary_edges[0][0]
    ordered = [start]
    visited = {start}
    current = start
    while True:
        found_next = False
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                ordered.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                found_next = True
                break
        if not found_next:
            break

    return np.array(mesh.vertices[ordered], dtype=np.float64)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Snapshot of training progress."""

    iteration: int = 0
    current_loss: float = 0.0
    surface_loss: float = 0.0
    total_loss: float = 0.0
    lr: float = 0.0
    elapsed_sec: float = 0.0
    losses: list[float] = field(default_factory=list)


class DeepCurrentsTrainer:
    """Trains a DeepCurrentsModel to reconstruct a target surface.

    The training procedure minimizes two complementary losses:

    1. Current loss (mass norm): encourages j(x) to align with the target surface.
       Under the anisotropic metric B_x = (I - n*n^T), patches aligned with the
       target surface cost nothing; misaligned patches are penalized.

    2. Surface loss (hinge): encourages f to have a jump across the target surface,
       complementing the current loss which only controls orientation.

    For minimal surfaces (no target data), only the Euclidean current loss is used.
    """

    def __init__(
        self,
        model: DeepCurrentsModel,
        target: PreparedTarget,
        config: DeepCurrentsConfig,
    ):
        self.model = model
        self.target = target
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay
        )
        self.state = TrainingState()

        self._target_mesh_trimesh = trimesh.Trimesh(
            vertices=target.vertices.cpu().numpy(),
            faces=target.faces.cpu().numpy(),
            process=False,
        )

    def train(self, progress_callback=None) -> TrainingState:
        """Run the full training loop.

        Parameters
        ----------
        progress_callback : optional callable(TrainingState) invoked every log_every steps

        Returns
        -------
        Final TrainingState with loss history
        """
        self.model.train()
        device = self.config.device
        t_start = time.perf_counter()

        logger.info(
            f"DeepCurrents training: {self.config.n_iterations} iterations, "
            f"lr={self.config.lr}, device={device}"
        )

        for it in range(1, self.config.n_iterations + 1):
            self.optimizer.zero_grad()

            # --- Sample points ---
            n_u = self.config.n_samples_uniform
            n_s = self.config.n_samples_surface

            x_uniform = torch.rand(n_u, 3, device=device) * 2 - 1
            x_uniform.requires_grad_(True)

            # Surface samples with inside/outside pairs
            idx = torch.randint(0, len(self.target.surface_points), (n_s,))
            x_surf = self.target.surface_points[idx]
            n_surf = self.target.surface_normals[idx]

            eps = torch.rand(n_s, 1, device=device) * (
                self.config.surface_loss_eps_max - self.config.surface_loss_eps_min
            ) + self.config.surface_loss_eps_min

            x_inside = (x_surf - eps * n_surf).detach().requires_grad_(True)
            x_outside = (x_surf + eps * n_surf).detach().requires_grad_(True)

            x_all = torch.cat([x_uniform, x_inside, x_outside], dim=0)

            # --- Forward pass ---
            out = self.model(x_all)
            current_uniform = out["current"][:n_u]

            # --- Current loss with anisotropic metric ---
            x_uniform_np = x_uniform.detach().cpu().numpy()
            closest_pts, dists, face_idx = trimesh.proximity.closest_point(
                self._target_mesh_trimesh, x_uniform_np
            )
            closest_normals = torch.tensor(
                self._target_mesh_trimesh.face_normals[face_idx],
                dtype=torch.float32, device=device
            )

            bdry_dists = compute_boundary_weights(
                x_uniform.detach(), self.target.boundary_verts, sigma=1e10
            )
            dist_to_bdry_sq = -2.0 * self.config.boundary_weight_sigma ** 2 * torch.log(
                bdry_dists.clamp(min=1e-30)
            )
            is_boundary = torch.zeros(n_u, dtype=torch.bool, device=device)

            transformed = compute_anisotropic_metric(
                current_uniform, closest_normals, is_boundary
            )

            # Boundary weighting: first half uses Gaussian weights, second half uniform
            half = n_u // 2
            weights = torch.ones(n_u, device=device)
            weights[:half] = compute_boundary_weights(
                x_uniform[:half].detach(),
                self.target.boundary_verts,
                sigma=self.config.boundary_weight_sigma,
            )

            current_loss = (transformed.norm(p=2, dim=-1) * weights).sum() / weights.sum()

            # --- Surface loss (hinge) ---
            f_inside = out["f"][n_u : n_u + n_s]
            f_outside = out["f"][n_u + n_s :]
            surface_loss = F.relu(
                f_outside - f_inside + self.config.surface_loss_delta
            ).mean()

            # --- Total loss ---
            total_loss = current_loss + surface_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # --- Logging ---
            self.state.iteration = it
            self.state.current_loss = current_loss.item()
            self.state.surface_loss = surface_loss.item()
            self.state.total_loss = total_loss.item()
            self.state.lr = self.scheduler.get_last_lr()[0]
            self.state.elapsed_sec = time.perf_counter() - t_start
            self.state.losses.append(total_loss.item())

            if it % self.config.log_every == 0 or it == 1:
                logger.info(
                    f"[{it:>6d}/{self.config.n_iterations}] "
                    f"loss={total_loss.item():.6f} "
                    f"(current={current_loss.item():.6f}, "
                    f"surface={surface_loss.item():.6f}) "
                    f"lr={self.state.lr:.2e} "
                    f"elapsed={self.state.elapsed_sec:.1f}s"
                )
                if progress_callback is not None:
                    progress_callback(self.state)

        logger.info(
            f"Training complete: final_loss={self.state.total_loss:.6f}, "
            f"elapsed={self.state.elapsed_sec:.1f}s"
        )
        return self.state


# ---------------------------------------------------------------------------
# Mesh extraction via marching cubes
# ---------------------------------------------------------------------------

def extract_mesh(
    model: DeepCurrentsModel,
    target: PreparedTarget,
    config: DeepCurrentsConfig,
    mode: Literal["auto", "mean_boundary"] = "mean_boundary",
) -> trimesh.Trimesh:
    """Extract a triangle mesh from the learned implicit current.

    Procedure (from Palmer et al. Section 5.2):
    1. Compute the average value s of f on the boundary curve(s)
    2. Run marching cubes on f(x) = s to get a closed isosurface
    3. Prune vertices where |j(x)| < threshold (removes the "phantom" surface
       that completes the closure)

    The result is an open surface with the prescribed boundary.

    Parameters
    ----------
    model : trained DeepCurrentsModel
    target : PreparedTarget with normalization info
    config : DeepCurrentsConfig
    mode : isovalue selection strategy
    """
    from skimage.measure import marching_cubes

    model.eval()
    device = config.device
    res = config.marching_cubes_resolution

    logger.info(f"Extracting mesh: resolution={res}, mode={mode}")

    # Determine isovalue from boundary
    with torch.no_grad():
        bdry_flat = target.boundary_verts.reshape(-1, 3)
        f_bdry = model.evaluate_f(bdry_flat)
        isovalue = f_bdry.mean().item()
    logger.info(f"Isovalue from boundary: {isovalue:.6f}")

    # Evaluate f on a regular grid
    lin = torch.linspace(config.domain_min, config.domain_max, res, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    f_vals = []
    batch_size = 32768
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            batch = grid_pts[i : i + batch_size]
            f_batch = model.evaluate_f(batch)
            f_vals.append(f_batch.cpu())

    f_grid = torch.cat(f_vals, dim=0).reshape(res, res, res).numpy()

    try:
        mc_verts, mc_faces, mc_normals, _ = marching_cubes(
            f_grid, level=isovalue,
            spacing=(2.0 / res, 2.0 / res, 2.0 / res),
        )
    except ValueError:
        logger.warning("Marching cubes found no surface at the computed isovalue")
        return trimesh.Trimesh()

    mc_verts = mc_verts + config.domain_min

    logger.info(
        f"Marching cubes: {len(mc_verts)} vertices, {len(mc_faces)} faces "
        f"before pruning"
    )

    # Prune low-current vertices
    mc_verts_t = torch.tensor(mc_verts, dtype=torch.float32, device=device)
    mc_verts_t.requires_grad_(True)

    current_norms = []
    with torch.enable_grad():
        for i in range(0, len(mc_verts_t), batch_size):
            batch = mc_verts_t[i : i + batch_size]
            if not batch.requires_grad:
                batch.requires_grad_(True)
            out = model(batch)
            norms = out["current"].detach().norm(p=2, dim=-1)
            current_norms.append(norms.cpu())

    current_norms = torch.cat(current_norms, dim=0).numpy()

    keep_mask = current_norms >= config.current_prune_threshold
    n_pruned = (~keep_mask).sum()
    logger.info(
        f"Pruning: removing {n_pruned}/{len(mc_verts)} vertices "
        f"with |j| < {config.current_prune_threshold}"
    )

    if keep_mask.sum() == 0:
        logger.warning("All vertices pruned. Try reducing current_prune_threshold.")
        return trimesh.Trimesh()

    old_to_new = np.full(len(mc_verts), -1, dtype=np.int64)
    new_idx = np.arange(keep_mask.sum())
    old_to_new[keep_mask] = new_idx

    new_verts = mc_verts[keep_mask]
    new_faces = []
    for face in mc_faces:
        mapped = old_to_new[face]
        if np.all(mapped >= 0):
            new_faces.append(mapped)

    if not new_faces:
        logger.warning("No valid faces after pruning.")
        return trimesh.Trimesh()

    new_faces = np.array(new_faces)

    # Denormalize to original coordinates
    new_verts = new_verts * target.normalization_scale + target.normalization_center

    result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    logger.info(
        f"Final mesh: {len(result.vertices)} vertices, {len(result.faces)} faces"
    )
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class DeepCurrentsResult:
    """Result container matching the Poisson reconstruction interface."""

    mesh: trimesh.Trimesh
    training_state: TrainingState
    elapsed_ms: float
    num_input_points: int
    num_output_vertices: int
    config: DeepCurrentsConfig


def deep_currents_reconstruct(
    points: NDArray[np.float64],
    normals: NDArray[np.float64] | None = None,
    boundary_vertices: NDArray[np.float64] | None = None,
    mesh_for_target: trimesh.Trimesh | None = None,
    config: DeepCurrentsConfig | None = None,
    progress_callback=None,
) -> DeepCurrentsResult:
    """Reconstruct a surface with explicit boundary using DeepCurrents.

    This is the main entry point, designed to parallel poisson_reconstruct()
    in interface style while offering boundary-aware neural reconstruction.

    Workflow:
    1. If mesh_for_target is provided, use it directly as the target geometry.
       Otherwise, run a quick Poisson reconstruction from the point cloud to
       create a target mesh (the metric needs face normals and closest-point
       queries).
    2. Normalize the target to [-0.5, 0.5]^3.
    3. Train the DeepCurrents model.
    4. Extract and denormalize the mesh via marching cubes.

    Parameters
    ----------
    points : (N, 3) input point cloud
    normals : (N, 3) point normals (estimated if None)
    boundary_vertices : (V, 3) ordered boundary loop. If None, extracted from
                        the target mesh boundary.
    mesh_for_target : optional trimesh.Trimesh to use as the target surface
                      for the metric. If None, Poisson is used internally.
    config : hyperparameters (uses defaults if None)
    progress_callback : optional callable(TrainingState)

    Returns
    -------
    DeepCurrentsResult with the reconstructed mesh and training history
    """
    t0 = time.perf_counter()

    if config is None:
        config = DeepCurrentsConfig()

    logger.info(
        f"DeepCurrents reconstruction: {len(points)} input points, "
        f"device={config.device}"
    )

    # Step 1: get a target mesh
    if mesh_for_target is None:
        from livemesh.reconstruction.poisson import poisson_reconstruct

        logger.info("No target mesh provided; running Poisson as initialization")
        poisson_result = poisson_reconstruct(points, normals=normals, depth=7)
        mesh_for_target = poisson_result.mesh

    # Step 2: prepare target data
    target = prepare_target(
        mesh_for_target,
        boundary_vertices=boundary_vertices,
        n_surface_samples=max(config.n_samples_surface * 5, 20_000),
        device=config.device,
    )

    # Step 3: build and train model
    model = DeepCurrentsModel(config, target.boundary_verts).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    trainer = DeepCurrentsTrainer(model, target, config)
    state = trainer.train(progress_callback=progress_callback)

    # Step 4: extract mesh
    result_mesh = extract_mesh(model, target, config)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"DeepCurrents complete: {len(result_mesh.vertices)} vertices, "
        f"elapsed={elapsed_ms:.0f} ms"
    )

    return DeepCurrentsResult(
        mesh=result_mesh,
        training_state=state,
        elapsed_ms=elapsed_ms,
        num_input_points=len(points),
        num_output_vertices=len(result_mesh.vertices),
        config=config,
    )


# ---------------------------------------------------------------------------
# Minimal surface computation (no target data)
# ---------------------------------------------------------------------------

def compute_minimal_surface(
    boundary_verts: NDArray[np.float64],
    config: DeepCurrentsConfig | None = None,
    progress_callback=None,
) -> trimesh.Trimesh:
    """Compute the minimal surface (soap film) spanning a boundary curve.

    This is the "pure" Plateau problem: minimize the mass norm under the
    Euclidean metric, without any target surface data. The result is the
    surface of least area bounded by the given curve.

    Parameters
    ----------
    boundary_verts : (V, 3) ordered vertices of a closed polygonal boundary
    config : hyperparameters
    progress_callback : optional callable(TrainingState)

    Returns
    -------
    trimesh.Trimesh of the minimal surface
    """
    if config is None:
        config = DeepCurrentsConfig(
            lr=5e-4,
            n_iterations=50_000,
            lr_decay_every=10_000,
            n_samples_uniform=4096,
        )

    device = config.device

    center = (boundary_verts.max(0) + boundary_verts.min(0)) / 2.0
    scale = np.abs(boundary_verts - center).max() * 2.2
    bdry_norm = (boundary_verts - center) / scale
    bdry_t = torch.tensor(bdry_norm, dtype=torch.float32, device=device)

    model = DeepCurrentsModel(config, bdry_t).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay
    )

    logger.info(
        f"Minimal surface: {config.n_iterations} iterations, "
        f"boundary={len(boundary_verts)} vertices"
    )

    state = TrainingState()
    t_start = time.perf_counter()

    for it in range(1, config.n_iterations + 1):
        optimizer.zero_grad()

        x = torch.rand(config.n_samples_uniform, 3, device=device) * 2 - 1
        x.requires_grad_(True)

        out = model(x)
        loss = out["current"].norm(p=2, dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        state.iteration = it
        state.total_loss = loss.item()
        state.elapsed_sec = time.perf_counter() - t_start

        if it % config.log_every == 0 or it == 1:
            logger.info(
                f"[{it:>6d}/{config.n_iterations}] mass_norm={loss.item():.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if progress_callback is not None:
                progress_callback(state)

    # Create a dummy target for mesh extraction
    target_dummy = PreparedTarget(
        vertices=bdry_t.unsqueeze(0),
        faces=torch.zeros(0, 3, dtype=torch.long, device=device),
        face_normals=torch.zeros(0, 3, device=device),
        boundary_verts=bdry_t,
        boundary_edges=torch.zeros(0, 2, 3, device=device),
        surface_points=torch.zeros(0, 3, device=device),
        surface_normals=torch.zeros(0, 3, device=device),
        normalization_center=center,
        normalization_scale=scale,
    )

    mesh = extract_mesh(model, target_dummy, config)
    logger.info(f"Minimal surface: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh
