"""Tests for synthetic surface generators."""

import numpy as np
import pytest

from livemesh.data.synthetic import (
    add_noise,
    add_occlusion,
    cylinder_patch,
    flat_plane,
    saddle_surface,
    sphere_cap,
    wound_crater,
)


class TestSphereCapSurface:
    def test_creates_valid_mesh(self):
        mesh = sphere_cap(radius=50.0, cap_angle_deg=60.0, resolution=32)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.is_watertight or len(mesh.faces) > 10

    def test_vertices_on_sphere(self):
        r = 50.0
        mesh = sphere_cap(radius=r, resolution=32)
        dists = np.linalg.norm(mesh.vertices, axis=1)
        np.testing.assert_allclose(dists, r, atol=1e-10)

    def test_resolution_affects_density(self):
        low = sphere_cap(resolution=16)
        high = sphere_cap(resolution=64)
        assert len(high.vertices) > len(low.vertices)


class TestSaddleSurface:
    def test_creates_valid_mesh(self):
        mesh = saddle_surface(size=40.0, curvature=0.01)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_saddle_shape(self):
        mesh = saddle_surface(size=40.0, curvature=0.01)
        z = mesh.vertices[:, 2]
        assert np.min(z) < 0  # concave regions
        assert np.max(z) > 0  # convex regions


class TestWoundCrater:
    def test_creates_valid_mesh(self):
        mesh = wound_crater(outer_radius=40.0, depth=8.0)
        assert len(mesh.vertices) > 0

    def test_has_depression(self):
        mesh = wound_crater(depth=8.0)
        z = mesh.vertices[:, 2]
        assert np.min(z) < -5.0  # crater goes down


class TestCylinderPatch:
    def test_creates_valid_mesh(self):
        mesh = cylinder_patch(radius=50.0, arc_angle_deg=90.0)
        assert len(mesh.vertices) > 0

    def test_vertices_on_cylinder(self):
        r = 50.0
        mesh = cylinder_patch(radius=r, resolution=32)
        yz_dist = np.sqrt(mesh.vertices[:, 1] ** 2 + mesh.vertices[:, 2] ** 2)
        np.testing.assert_allclose(yz_dist, r, atol=1e-10)


class TestFlatPlane:
    def test_all_z_zero(self):
        mesh = flat_plane(size=60.0)
        np.testing.assert_allclose(mesh.vertices[:, 2], 0.0, atol=1e-10)


class TestNoise:
    def test_adds_noise(self):
        mesh = sphere_cap(resolution=16)
        noisy = add_noise(mesh, sigma=0.5)
        assert noisy.shape == mesh.vertices.shape
        diff = np.linalg.norm(noisy - mesh.vertices, axis=1)
        assert np.mean(diff) > 0.1

    def test_reproducible(self):
        mesh = sphere_cap(resolution=16)
        a = add_noise(mesh, sigma=1.0, rng=np.random.default_rng(42))
        b = add_noise(mesh, sigma=1.0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestOcclusion:
    def test_removes_points(self):
        mesh = sphere_cap(resolution=32)
        pts = np.array(mesh.vertices)
        occluded = add_occlusion(pts, fraction=0.2)
        assert len(occluded) < len(pts)
        assert len(occluded) > len(pts) * 0.5  # shouldn't remove too many
