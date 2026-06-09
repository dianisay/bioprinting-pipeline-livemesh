# DeepCurrents: Neural Implicit Surfaces with Boundary Control

Implementation of **Palmer, Smirnov, Wang, Chern & Solomon** (CVPR 2022)
for the LiveMesh bioprinting pipeline.

> **Why this matters for bioprinting:** Standard implicit methods (Poisson, SDFs)
> reconstruct *closed* surfaces and cannot enforce that the reconstructed wound
> surface ends precisely at the segmented wound boundary. DeepCurrents produces
> an *open* surface whose boundary is an explicit, user-controlled curve.
> This means geodesic toolpaths start and stop exactly at the wound edge.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [API Reference](#4-api-reference)
5. [Training Details](#5-training-details)
6. [Mesh Extraction](#6-mesh-extraction)
7. [Bioprinting Integration](#7-bioprinting-integration)
8. [Minimal Surfaces](#8-minimal-surfaces)
9. [Benchmarking](#9-benchmarking)
10. [Configuration Reference](#10-configuration-reference)
11. [Design Decisions](#11-design-decisions)

---

## 1. Mathematical Foundation

### The Problem

Given a boundary curve \(\Gamma\) (e.g., the wound contour from segmentation)
and a target surface \(\Sigma\) (from a depth camera), find a surface that:
- interpolates \(\Gamma\) exactly as its boundary
- matches the geometry of \(\Sigma\)

### Currents from Geometric Measure Theory

A 2-current is a generalized surface, defined by its action on differential
2-forms. A surface \(\Sigma\) with boundary \(\partial\Sigma = \Gamma\) induces
a current via integration:

```
[Sigma](zeta) = integral_Sigma zeta
```

The **mass norm** generalizes area:

```
M(T) = sup { T(zeta) : |zeta_x| <= 1  for all x }
```

### Hodge Decomposition

The space of currents satisfying the boundary constraint is parameterized as:

```
omega = df + alpha_Gamma
```

where:
- `f_theta : R^3 -> R` is a neural network (the learned part)
- `alpha_Gamma` is the **Biot-Savart field** of the boundary (computed in closed form)
- `df = grad f` is obtained via automatic differentiation

The **current vector** at any point `x` is:

```
j(x) = grad_x f(x) + alpha(x)
```

This vector points in the surface normal direction wherever the surface exists.

### Biot-Savart Field

For a polygonal boundary curve with segments `(v_i, v_{i+1})`:

```
alpha(x) = sum_i  [t_hat_i . (r_hat1_i - r_hat0_i)] * (t_hat_i x r0_i)
                   / |t_hat_i x r0_i|^2
```

where `t_hat_i` is the unit tangent of segment `i`, and `r0_i = v_i - x`.
This is the magnetic field of a current-carrying wire loop, computed analytically.

### Loss Functions

**Current loss** (anisotropic mass norm):

```
L_curr = E_{x ~ Uniform} [ |B_x * j(x)| ]
```

where `B_x = I - n*n^T` projects out the component aligned with the
target surface normal `n` at the closest point on `Sigma`.

**Surface loss** (hinge):

```
L_surf = E_{x ~ Sigma} [ max(0, delta - f(x - eps*n) + f(x + eps*n)) ]
```

This forces `f` to jump across the target surface.

**Total loss:**

```
L = L_curr + L_surf
```

---

## 2. Architecture

```
Input: query point x in R^3
         |
    [Random Fourier Features]
    gamma(x) = [sin(2*pi*B*x), cos(2*pi*B*x)]
    B ~ N(0, sigma^2),  output: R^2048
         |
    [MLP f_theta]
    4 hidden layers x 256 units, Softplus activation
    output: scalar f(x)
         |
    [Autograd]
    df = grad_x f(x)
         |
    [Biot-Savart] ----+
    alpha(x)          |
         |            |
    [Current: j = df + alpha]
         |
    [Anisotropic Metric B_x]
    B_x * j(x)
         |
    [Mass Norm Loss]
    |B_x * j(x)|
```

**Key design choices** (from the ablation study in the paper):

| Choice | Why |
|--------|-----|
| **Softplus** (not ReLU) | ReLU has zero second derivative, which blurs the current |
| **Random Fourier Features** | Without them, the MLP cannot learn high-frequency detail |
| **Surface loss** | The metric alone is insufficient for convergence |
| **Boundary weighting** | Gaussian weight `exp(-d^2/2*sigma^2)` sharpens results near the boundary |
| **Biot-Savart scale 1/1000** | Balances gradient magnitudes with network initialization |

---

## 3. Quick Start

### Surface Reconstruction (30-second version)

```python
from livemesh.data.synthetic import sphere_cap, add_noise
from livemesh.reconstruction import deep_currents_reconstruct, DeepCurrentsConfig
from livemesh.utils.logging_config import setup_logging

setup_logging("INFO")

# Generate a test surface with boundary (open mesh)
mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=32)
points = add_noise(mesh, sigma=0.3)

# Reconstruct with DeepCurrents
result = deep_currents_reconstruct(
    points=points,
    mesh_for_target=mesh,
    config=DeepCurrentsConfig(n_iterations=5000, device="cpu"),
)

print(f"Reconstructed: {result.num_output_vertices} vertices")
print(f"Time: {result.elapsed_ms:.0f} ms")
result.mesh.export("reconstructed.ply")
```

### Minimal Surface (Soap Film)

```python
import numpy as np
from livemesh.reconstruction import compute_minimal_surface, DeepCurrentsConfig

# Trefoil knot boundary
t = np.linspace(0, 2*np.pi, 100, endpoint=False)
boundary = np.column_stack([
    np.sin(t) + 2*np.sin(2*t),
    np.cos(t) - 2*np.cos(2*t),
    -np.sin(3*t),
]) / 6.0

mesh = compute_minimal_surface(
    boundary,
    config=DeepCurrentsConfig(n_iterations=50000, device="cuda"),
)
mesh.export("trefoil_minimal.ply")
```

---

## 4. API Reference

### `deep_currents_reconstruct()`

Main entry point for surface reconstruction.

```python
def deep_currents_reconstruct(
    points: NDArray,                          # (N, 3) input point cloud
    normals: NDArray | None = None,           # (N, 3) point normals
    boundary_vertices: NDArray | None = None, # (V, 3) ordered boundary loop
    mesh_for_target: Trimesh | None = None,   # target mesh for the metric
    config: DeepCurrentsConfig | None = None, # hyperparameters
    progress_callback = None,                 # callable(TrainingState)
) -> DeepCurrentsResult
```

**Returns** `DeepCurrentsResult`:
| Field | Type | Description |
|-------|------|-------------|
| `mesh` | `trimesh.Trimesh` | Reconstructed surface |
| `training_state` | `TrainingState` | Loss history and timing |
| `elapsed_ms` | `float` | Total wall time |
| `num_input_points` | `int` | Input point count |
| `num_output_vertices` | `int` | Output vertex count |
| `config` | `DeepCurrentsConfig` | Configuration used |

### `compute_minimal_surface()`

Compute the minimal surface spanning a boundary curve (Plateau's problem).

```python
def compute_minimal_surface(
    boundary_verts: NDArray,                  # (V, 3) closed polyline
    config: DeepCurrentsConfig | None = None,
    progress_callback = None,
) -> trimesh.Trimesh
```

### `DeepCurrentsModel`

The neural network module (for advanced usage).

```python
model = DeepCurrentsModel(config, boundary_verts_tensor)
out = model(x)  # x requires grad
# out["f"]       : (N, 1)  scalar field
# out["current"] : (N, 3)  current vector j = df + alpha
```

### `biot_savart_3d()`

Closed-form Biot-Savart field of a polygonal boundary.

```python
alpha = biot_savart_3d(x, boundary_verts, scale=1e-3)
# x:             (M, 3)       query points
# boundary_verts: (L, V, 3)   L loops, V vertices each
# alpha:         (M, 3)       vector field
```

### `prepare_target()`

Normalize a mesh and extract training data.

```python
target = prepare_target(mesh, boundary_vertices=None, n_surface_samples=10000)
```

### `extract_mesh()`

Marching cubes extraction from a trained model.

```python
mesh = extract_mesh(model, target, config)
```

---

## 5. Training Details

### Default Hyperparameters (Reconstruction)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | Adam optimizer |
| LR decay | 0.6 every 2000 steps | Exponential schedule |
| Iterations | 10,000 | ~4 min on RTX 3090 |
| Uniform samples | 4,000/iter | For mass norm loss |
| Surface samples | 4,000/iter | For hinge loss |
| RFF dimension | 2,048 | sin/cos features |
| RFF sigma | 2.0 | Bandwidth (variance=4) |
| Hidden layers | 4 x 256 | Softplus activations |
| Surface delta | 0.01 | Hinge margin |
| Epsilon range | [0.001, 0.02] | Inside/outside offset |
| Boundary sigma | 0.1 | Gaussian weight falloff |
| Biot-Savart scale | 1e-3 | Normalization |

### Training Loop (pseudocode)

```
for each iteration:
    # 1. Sample 4000 uniform points in [-1,1]^3
    # 2. Sample 4000 surface points with inside/outside pairs

    # 3. Forward pass: compute j(x) = grad f(x) + alpha(x)

    # 4. Current loss:
    #    - Find closest face normal for each uniform point (via trimesh)
    #    - Apply metric: B_x = (I - n*n^T)
    #    - Compute boundary weights (Gaussian)
    #    - L_curr = weighted mean |B_x * j(x)|

    # 5. Surface loss:
    #    - L_surf = mean(max(0, f_outside - f_inside + delta))

    # 6. Total: L = L_curr + L_surf
    # 7. Backprop + Adam step + LR decay
```

### Training Output

```
[     1/10000] loss=0.183542 (current=0.183478, surface=0.000064) lr=1.00e-03 elapsed=0.3s
[   500/10000] loss=0.042187 (current=0.041932, surface=0.000255) lr=9.22e-04 elapsed=28.1s
[  1000/10000] loss=0.031456 (current=0.031201, surface=0.000255) lr=8.50e-04 elapsed=56.4s
...
[ 10000/10000] loss=0.008923 (current=0.008670, surface=0.000253) lr=2.18e-04 elapsed=237.5s
```

---

## 6. Mesh Extraction

After training, a triangle mesh is extracted via marching cubes:

1. **Isovalue**: Average `f` over the boundary curve(s). Since `f` is constant
   along the normal direction near the surface, averaging on the boundary
   gives a consistent isovalue.

2. **Marching cubes**: Extract the level set `f(x) = s` on a regular grid
   (default 128^3). This produces a *closed* isosurface.

3. **Pruning**: Remove vertices where `|j(x)| < threshold` (default 5e-3).
   These are "phantom" vertices from the closure that don't correspond to
   actual surface. The result is an open surface with the correct boundary.

4. **Denormalization**: Map from [-0.5, 0.5]^3 back to original coordinates.

---

## 7. Bioprinting Integration

### Pipeline Flow

```
[Depth Camera] --> point cloud
       |
[U-Net Segmentation] --> wound boundary (ordered polygon)
       |
       v
[DeepCurrents]
  - boundary = wound contour
  - target = point cloud + Poisson initialization
  - output = open mesh with boundary at wound edge
       |
[Geodesic Toolpath] --> nozzle trajectories on wound surface
       |
[Robot Controller] --> extrusion commands
```

### Usage in the LiveMesh Pipeline

```python
from livemesh.pipeline.orchestrator import LiveMeshPipeline

pipeline = LiveMeshPipeline(config_path="configs/default.yaml")

# The orchestrator calls deep_currents_reconstruct automatically when
# reconstruction_method="deep_currents" is set in the config.
result = pipeline.run(
    image=depth_image,
    point_cloud=points,
    reconstruction_method="deep_currents",
)
```

### Why DeepCurrents Instead of Poisson?

| Feature | Poisson | DeepCurrents |
|---------|---------|--------------|
| Boundary control | None (closed surface) | Explicit boundary curves |
| Open surfaces | Requires post-trimming | Native support |
| Resolution | Fixed octree | Arbitrary (neural) |
| Topology | Single connected component | Any topology |
| Speed | ~100 ms | ~4 min (training) |
| Accuracy near boundary | Artifacts | Sharp edges |

**When to use each:**
- **Poisson**: Real-time feedback during printing (fast, good enough)
- **DeepCurrents**: Pre-print surface preparation (accurate, boundary-aware)

---

## 8. Minimal Surfaces

DeepCurrents can also solve Plateau's problem: find the surface of minimum
area bounded by a given curve. This is useful for:

- **Initial surface estimate**: Before fitting to data, compute the minimal
  surface as a starting point
- **Testing**: Verify the implementation on known analytical solutions
- **Visualization**: Generate soap-film-like surfaces for presentations

### Classic Test Cases

```python
import numpy as np
from livemesh.reconstruction import compute_minimal_surface, DeepCurrentsConfig

t = np.linspace(0, 2*np.pi, 100, endpoint=False)

# Trefoil knot
trefoil = np.column_stack([
    np.sin(t) + 2*np.sin(2*t),
    np.cos(t) - 2*np.cos(2*t),
    -np.sin(3*t),
]) / 6.0

# Hopf link (two interlocking circles)
hopf_1 = np.column_stack([0.6*np.cos(t) - 0.3, 0.6*np.sin(t), np.zeros_like(t)])
hopf_2 = np.column_stack([0.6*np.cos(t) + 0.3, np.zeros_like(t), 0.6*np.sin(t)])
```

---

## 9. Benchmarking

DeepCurrents integrates with the existing benchmarking system:

```python
from livemesh.reconstruction import (
    deep_currents_reconstruct,
    poisson_reconstruct,
    benchmark_reconstruction,
    DeepCurrentsConfig,
)
from livemesh.data.synthetic import sphere_cap, add_noise

# Ground truth
gt_mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=64)
noisy_pts = add_noise(gt_mesh, sigma=0.5)
normals = gt_mesh.vertex_normals

# Poisson baseline
poisson = poisson_reconstruct(noisy_pts, normals=normals, depth=8)
poisson_bench = benchmark_reconstruction(poisson.mesh, gt_mesh, poisson.elapsed_ms)

# DeepCurrents
dc = deep_currents_reconstruct(
    noisy_pts, normals=normals, mesh_for_target=gt_mesh,
    config=DeepCurrentsConfig(n_iterations=10000, device="cuda"),
)
dc_bench = benchmark_reconstruction(dc.mesh, gt_mesh, dc.elapsed_ms)

print(f"Poisson:       Hausdorff={poisson_bench.hausdorff_mm:.3f} mm, "
      f"Mean={poisson_bench.mean_distance_mm:.3f} mm")
print(f"DeepCurrents:  Hausdorff={dc_bench.hausdorff_mm:.3f} mm, "
      f"Mean={dc_bench.mean_distance_mm:.3f} mm")
```

### Expected Performance (from Palmer et al.)

| Shape | Method | Unidirectional Chamfer |
|-------|--------|----------------------|
| Head | NCP (Venkatesh 2021) | 0.0049 |
| Head | DeepCurrents | **0.0010** |
| Hand | NCP | 0.0045 |
| Hand | DeepCurrents | **0.0011** |
| Torso | NCP | 0.0049 |
| Torso | DeepCurrents | **0.00092** |
| Foot | NCP | 0.0055 |
| Foot | DeepCurrents | **0.00092** |

---

## 10. Configuration Reference

All hyperparameters are in `DeepCurrentsConfig`:

```python
@dataclass
class DeepCurrentsConfig:
    # Network
    rff_dim: int = 2048           # Random Fourier Feature dimension
    rff_sigma: float = 2.0        # RFF bandwidth (variance = sigma^2)
    hidden_dim: int = 256         # MLP hidden layer width
    num_hidden_layers: int = 4    # Number of hidden layers

    # Training
    lr: float = 1e-3              # Adam learning rate
    lr_decay: float = 0.6         # LR multiplier
    lr_decay_every: int = 2000    # LR decay interval (iterations)
    n_iterations: int = 10_000    # Total training iterations
    n_samples_uniform: int = 4000 # Uniform samples per iteration
    n_samples_surface: int = 4000 # Surface samples per iteration

    # Loss
    surface_loss_delta: float = 0.01    # Hinge margin
    surface_loss_eps_min: float = 0.001 # Min inside/outside offset
    surface_loss_eps_max: float = 0.02  # Max inside/outside offset
    boundary_weight_sigma: float = 0.1  # Gaussian boundary weight falloff
    biot_savart_scale: float = 1e-3     # Alpha normalization

    # Meshing
    marching_cubes_resolution: int = 128  # Grid resolution for extraction
    current_prune_threshold: float = 5e-3 # Remove vertices with |j| below this

    # Domain
    domain_min: float = -1.0
    domain_max: float = 1.0

    # Device
    device: str = "cuda"          # or "cpu"

    # Logging
    log_every: int = 500          # Print loss every N iterations
```

### Tuning Guide

| Scenario | Adjustments |
|----------|-------------|
| **Faster training** | Reduce `n_iterations` to 5000, `n_samples_*` to 2000 |
| **Higher quality** | Increase `n_iterations` to 20000, `marching_cubes_resolution` to 256 |
| **Small features** | Increase `rff_sigma` to 4.0, `hidden_dim` to 512 |
| **CPU-only** | Set `device="cpu"`, reduce samples |
| **Noisy boundary** | Increase `boundary_weight_sigma` to 0.2 |
| **Over-pruning** | Reduce `current_prune_threshold` to 1e-3 |

---

## 11. Design Decisions

### No PyTorch3D Dependency

The original implementation uses PyTorch3D's CUDA kernels for point-to-face and
point-to-edge distance computation. We replace these with:

- **trimesh.proximity.closest_point()** for point-to-face distances
  (runs on CPU, called once per iteration outside the gradient tape)
- **Pure PyTorch edge projection** for boundary weights

This makes the code portable: it runs on any machine with PyTorch + trimesh,
no CUDA compilation required. The performance cost is acceptable because the
metric computation is not the bottleneck (the MLP forward+backward is).

### Poisson Initialization

When no target mesh is provided, we run a quick Poisson reconstruction first
to create the target geometry for the metric. This bootstrapping step takes
~100ms and provides face normals for the anisotropic metric. The DeepCurrents
model then refines the surface while adding boundary control.

### Boundary Extraction

If no boundary is explicitly provided, we extract boundary loops automatically
from the mesh's non-manifold edges. This works well for open meshes (like
segmented wound surfaces) but fails for closed meshes. For closed meshes,
use Poisson reconstruction instead.

---

## References

1. Palmer, Smirnov, Wang, Chern & Solomon.
   *DeepCurrents: Learning Implicit Representations of Shapes with Boundaries.*
   CVPR 2022, pp. 18665-18675.

2. Tancik et al. *Fourier Features Let Networks Learn High Frequency
   Functions in Low Dimensional Domains.* NeurIPS 2020.

3. Kazhdan & Hoppe. *Screened Poisson Surface Reconstruction.*
   ACM ToG 32(3), 2013.

4. Original code: [github.com/dmsm/DeepCurrents](https://github.com/dmsm/DeepCurrents)
