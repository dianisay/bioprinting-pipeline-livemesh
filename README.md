# LiveMesh

**Real-time geometric surface reconstruction and geodesic toolpath generation for in-situ bioprinting on non-planar tissue.**

LiveMesh replaces the traditional planar-slicing approach in bioprinting with a geometry-aware pipeline: a depth camera captures the wound surface, the system reconstructs a smooth mesh in real time, and geodesic deposition paths are computed directly on the curved surface. A CNN visual controller closes the loop, updating the mesh and re-planning paths as the tissue deforms.

Developed as part of a research collaboration between [Tecnologico de Monterrey](https://tec.mx) and the [Geometric Data Processing Group](https://geometry.csail.mit.edu/) at MIT CSAIL (Prof. Justin Solomon).

---

## Pipeline

```
Depth camera  -->  Point cloud  -->  Surface reconstruction  -->  Geodesic toolpaths
      ^                                                                |
      |                                                                v
  CNN visual   <--  Deposition   <--  Robot waypoints  <--  Conformal mapping
  feedback          (hydrogel)        (G-code / ROS2)
```

## Modules

| Module | Description |
|--------|-------------|
| `livemesh.reconstruction` | Poisson baseline + implicit (DeepCurrents-style) surface reconstruction from point clouds |
| `livemesh.toolpath` | Geodesic and planar toolpath generation, conformal UV-to-XYZ mapping, coverage metrics |
| `livemesh.segmentation` | U-Net wound segmentation (ported from MATLAB), boundary extraction |
| `livemesh.data` | Synthetic surface generators (sphere cap, saddle, wound crater, cylinder patch) for testing |
| `livemesh.controller` | CNN-based closed-loop controller (integration point) |
| `livemesh.visualization` | Polyscope/matplotlib rendering for meshes, toolpaths, and figures |

## Quick start

```bash
# Clone
git clone https://github.com/dianisay/livemesh.git
cd livemesh

# Install
pip install -e ".[dev]"

# Run tests
pytest

# Generate a synthetic surface and compute geodesic toolpaths
python -c "
from livemesh.data import sphere_cap, add_noise
from livemesh.reconstruction import poisson_reconstruct, benchmark_reconstruction
from livemesh.toolpath import geodesic_toolpaths, coverage_uniformity

# Ground truth: curved wound surface
gt_mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=64)

# Simulate depth camera noise
points = add_noise(gt_mesh, sigma=0.5)

# Reconstruct
result = poisson_reconstruct(points, depth=8)
print(f'Reconstruction: {result.elapsed_ms:.1f} ms, {result.num_output_vertices} vertices')

# Benchmark
bench = benchmark_reconstruction(result.mesh, gt_mesh, elapsed_ms=result.elapsed_ms)
print(f'Hausdorff: {bench.hausdorff_mm:.2f} mm, Mean: {bench.mean_distance_mm:.3f} mm')

# Geodesic toolpaths
paths = geodesic_toolpaths(result.mesh, spacing_mm=2.0)
print(f'Generated {paths.num_paths} paths, total length {paths.total_length_mm:.1f} mm')

# Coverage analysis
cov = coverage_uniformity(result.mesh, paths.waypoints, nozzle_width_mm=1.5)
print(f'Coverage: {cov.coverage_fraction:.1%}, Uniformity: {cov.uniformity_score:.2f}')
"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Open3D >= 0.17
- trimesh >= 4.0
- potpourri3d >= 0.0.8

Full list in `pyproject.toml`.

## Project context

This software supports the doctoral thesis:

> *Convolutional Neural Network-Based Machine Learning for Three-Dimensional Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration*

Key publications:
- Wound segmentation and G-code generation pipeline (WoundPath-AI)
- Conformal trajectory generation for 8-DOF bioprinting systems
- Patent: Additively manufactured prosthetic sockets (MX, 2023)

## License

MIT
