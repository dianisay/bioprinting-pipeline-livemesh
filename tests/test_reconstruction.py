"""Tests for surface reconstruction."""

import numpy as np
import pytest

from livemesh.data.synthetic import add_noise, sphere_cap
from livemesh.reconstruction.benchmarks import benchmark_reconstruction
from livemesh.reconstruction.poisson import poisson_reconstruct


class TestPoissonReconstruction:
    @pytest.fixture
    def sphere_data(self):
        mesh = sphere_cap(radius=50.0, resolution=32)
        noisy_points = add_noise(mesh, sigma=0.3)
        return mesh, noisy_points

    def test_returns_valid_mesh(self, sphere_data):
        gt_mesh, points = sphere_data
        result = poisson_reconstruct(points, depth=6)
        assert result.mesh is not None
        assert len(result.mesh.vertices) > 0
        assert len(result.mesh.faces) > 0

    def test_reports_timing(self, sphere_data):
        _, points = sphere_data
        result = poisson_reconstruct(points, depth=6)
        assert result.elapsed_ms > 0

    def test_reports_counts(self, sphere_data):
        _, points = sphere_data
        result = poisson_reconstruct(points, depth=6)
        assert result.num_input_points == len(points)
        assert result.num_output_vertices > 0


class TestBenchmarks:
    def test_benchmark_identical_meshes(self):
        mesh = sphere_cap(radius=50.0, resolution=32)
        result = benchmark_reconstruction(mesh, mesh)
        assert result.hausdorff_mm < 1.0  # near-zero for identical
        assert result.mean_distance_mm < 0.5

    def test_benchmark_noisy_reconstruction(self):
        gt = sphere_cap(radius=50.0, resolution=32)
        points = add_noise(gt, sigma=0.5)
        rec = poisson_reconstruct(points, depth=6)
        result = benchmark_reconstruction(rec.mesh, gt, elapsed_ms=rec.elapsed_ms)
        assert result.hausdorff_mm < 20.0  # reasonable bound
        assert result.normal_deviation_deg < 45.0
