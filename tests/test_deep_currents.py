"""Tests for DeepCurrents neural implicit surface reconstruction."""

import numpy as np
import pytest
import torch

from livemesh.data.synthetic import sphere_cap, cylinder_patch, wound_crater
from livemesh.reconstruction.deep_currents import (
    RandomFourierFeatures,
    CurrentMLP,
    DeepCurrentsConfig,
    DeepCurrentsModel,
    biot_savart_3d,
    compute_anisotropic_metric,
    compute_boundary_weights,
    prepare_target,
    extract_mesh,
    deep_currents_reconstruct,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def simple_boundary(device):
    """Unit circle in the XY plane (32 vertices)."""
    t = torch.linspace(0, 2 * np.pi, 33)[:-1]
    verts = torch.stack([0.4 * torch.cos(t), 0.4 * torch.sin(t), torch.zeros_like(t)], dim=-1)
    return verts.to(device)


@pytest.fixture
def quick_config():
    return DeepCurrentsConfig(
        n_iterations=50,
        n_samples_uniform=256,
        n_samples_surface=256,
        marching_cubes_resolution=32,
        log_every=25,
        device="cpu",
    )


# ── Random Fourier Features ──────────────────────────────────────────────

class TestRandomFourierFeatures:
    def test_output_shape(self):
        rff = RandomFourierFeatures(d_in=3, d_out=2048, sigma=2.0)
        x = torch.randn(100, 3)
        out = rff(x)
        assert out.shape == (100, 2048)

    def test_deterministic(self):
        rff = RandomFourierFeatures(d_in=3, d_out=512, sigma=2.0)
        x = torch.randn(10, 3)
        assert torch.allclose(rff(x), rff(x))

    def test_output_bounded(self):
        rff = RandomFourierFeatures(d_in=3, d_out=1024, sigma=2.0)
        x = torch.randn(50, 3)
        out = rff(x)
        assert out.abs().max() <= 1.0 + 1e-6


# ── Current MLP ──────────────────────────────────────────────────────────

class TestCurrentMLP:
    def test_output_shape(self):
        mlp = CurrentMLP(d_in=2048, hidden_dim=64, num_layers=2)
        x = torch.randn(32, 2048)
        out = mlp(x)
        assert out.shape == (32, 1)

    def test_gradients_exist(self):
        mlp = CurrentMLP(d_in=128, hidden_dim=32, num_layers=2)
        x = torch.randn(8, 128, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 128)


# ── Biot-Savart ──────────────────────────────────────────────────────────

class TestBiotSavart:
    def test_output_shape(self, simple_boundary):
        x = torch.randn(50, 3)
        alpha = biot_savart_3d(x, simple_boundary)
        assert alpha.shape == (50, 3)

    def test_field_nonzero_inside(self, simple_boundary):
        x = torch.zeros(1, 3)
        alpha = biot_savart_3d(x, simple_boundary)
        assert alpha.norm().item() > 0

    def test_field_decays_far_away(self, simple_boundary):
        x_near = torch.tensor([[0.0, 0.0, 0.0]])
        x_far = torch.tensor([[10.0, 10.0, 10.0]])
        alpha_near = biot_savart_3d(x_near, simple_boundary).norm()
        alpha_far = biot_savart_3d(x_far, simple_boundary).norm()
        assert alpha_near > alpha_far

    def test_multiple_loops(self):
        loop1 = torch.randn(10, 3) * 0.3
        loop2 = torch.randn(10, 3) * 0.3 + 0.5
        bdry = torch.stack([loop1, loop2])
        x = torch.randn(20, 3)
        alpha = biot_savart_3d(x, bdry)
        assert alpha.shape == (20, 3)


# ── Anisotropic Metric ──────────────────────────────────────────────────

class TestAnisotropicMetric:
    def test_removes_normal_component(self):
        current = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        is_bdry = torch.tensor([False, False])
        result = compute_anisotropic_metric(current, normals, is_bdry)
        assert result[0].norm().item() < 1e-6
        assert torch.allclose(result[1], current[1], atol=1e-6)

    def test_boundary_points_unchanged(self):
        current = torch.randn(5, 3)
        normals = torch.randn(5, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        is_bdry = torch.ones(5, dtype=torch.bool)
        result = compute_anisotropic_metric(current, normals, is_bdry)
        assert torch.allclose(result, current)


# ── Boundary Weights ────────────────────────────────────────────────────

class TestBoundaryWeights:
    def test_closer_points_higher_weight(self, simple_boundary):
        x_close = torch.tensor([[0.4, 0.0, 0.0]])
        x_far = torch.tensor([[0.0, 0.0, 0.5]])
        w_close = compute_boundary_weights(x_close, simple_boundary, sigma=0.1)
        w_far = compute_boundary_weights(x_far, simple_boundary, sigma=0.1)
        assert w_close.item() > w_far.item()

    def test_weights_in_range(self, simple_boundary):
        x = torch.randn(100, 3)
        w = compute_boundary_weights(x, simple_boundary, sigma=0.1)
        assert (w >= 0).all()
        assert (w <= 1).all()


# ── Full Model ──────────────────────────────────────────────────────────

class TestDeepCurrentsModel:
    def test_forward_shapes(self, simple_boundary, quick_config):
        model = DeepCurrentsModel(quick_config, simple_boundary)
        x = torch.randn(16, 3, requires_grad=True)
        out = model(x, return_components=True)
        assert out["f"].shape == (16, 1)
        assert out["current"].shape == (16, 3)
        assert out["df"].shape == (16, 3)
        assert out["alpha"].shape == (16, 3)

    def test_evaluate_f_no_grad(self, simple_boundary, quick_config):
        model = DeepCurrentsModel(quick_config, simple_boundary)
        x = torch.randn(16, 3)
        f = model.evaluate_f(x)
        assert f.shape == (16, 1)

    def test_current_equals_df_plus_alpha(self, simple_boundary, quick_config):
        model = DeepCurrentsModel(quick_config, simple_boundary)
        x = torch.randn(8, 3, requires_grad=True)
        out = model(x, return_components=True)
        reconstructed = out["df"] + out["alpha"]
        assert torch.allclose(out["current"], reconstructed, atol=1e-5)


# ── Data Preparation ────────────────────────────────────────────────────

class TestPrepareTarget:
    def test_with_open_mesh(self):
        mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=16)
        target = prepare_target(mesh, n_surface_samples=100, device="cpu")
        assert target.vertices.shape[1] == 3
        assert target.boundary_verts.shape[1] == 3
        assert target.surface_points.shape == (100, 3)
        assert target.surface_normals.shape == (100, 3)
        # Normalized mesh should be in [-0.5, 0.5]^3
        assert target.vertices.abs().max() <= 0.55

    def test_with_explicit_boundary(self):
        mesh = sphere_cap(radius=30.0, cap_angle_deg=60.0, resolution=16)
        bdry = np.array(mesh.vertices[:10], dtype=np.float64)
        target = prepare_target(mesh, boundary_vertices=bdry, device="cpu")
        assert len(target.boundary_verts) == 10


# ── Integration Test ────────────────────────────────────────────────────

class TestDeepCurrentsReconstruct:
    @pytest.mark.slow
    def test_runs_end_to_end(self, quick_config):
        """Smoke test: runs the full pipeline on a small mesh."""
        mesh = sphere_cap(radius=30.0, cap_angle_deg=45.0, resolution=16)
        points = np.array(mesh.vertices, dtype=np.float64)
        normals = np.array(mesh.vertex_normals, dtype=np.float64)

        result = deep_currents_reconstruct(
            points=points,
            normals=normals,
            mesh_for_target=mesh,
            config=quick_config,
        )
        assert result.num_output_vertices >= 0
        assert result.elapsed_ms > 0
        assert len(result.training_state.losses) == quick_config.n_iterations
