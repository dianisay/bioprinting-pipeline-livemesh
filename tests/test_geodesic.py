"""Tests for geodesic toolpath generation."""

import numpy as np
import pytest

from livemesh.data.synthetic import cylinder_patch, flat_plane, sphere_cap
from livemesh.toolpath.coverage import coverage_uniformity
from livemesh.toolpath.geodesic import geodesic_toolpaths


class TestGeodesicToolpaths:
    def test_generates_paths_on_flat_surface(self):
        mesh = flat_plane(size=60.0, resolution=32)
        result = geodesic_toolpaths(mesh, spacing_mm=5.0)
        assert result.num_paths > 0
        assert len(result.waypoints) > 0
        assert result.total_length_mm > 0

    def test_generates_paths_on_curved_surface(self):
        mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=32)
        result = geodesic_toolpaths(mesh, spacing_mm=5.0)
        assert result.num_paths > 0
        assert len(result.waypoints) > 0

    def test_normals_are_unit_vectors(self):
        mesh = sphere_cap(radius=50.0, resolution=32)
        result = geodesic_toolpaths(mesh, spacing_mm=5.0)
        if len(result.normals) > 0:
            norms = np.linalg.norm(result.normals, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=0.1)

    def test_deposition_mask_matches_waypoints(self):
        mesh = flat_plane(size=60.0, resolution=32)
        result = geodesic_toolpaths(mesh, spacing_mm=5.0)
        if len(result.waypoints) > 0:
            assert len(result.is_deposition) == len(result.waypoints)

    def test_tighter_spacing_more_paths(self):
        mesh = flat_plane(size=60.0, resolution=32)
        wide = geodesic_toolpaths(mesh, spacing_mm=10.0)
        tight = geodesic_toolpaths(mesh, spacing_mm=3.0)
        assert tight.num_paths >= wide.num_paths


class TestCoverage:
    def test_coverage_fraction_bounded(self):
        mesh = flat_plane(size=40.0, resolution=32)
        result = geodesic_toolpaths(mesh, spacing_mm=3.0)
        if len(result.waypoints) > 0:
            cov = coverage_uniformity(mesh, result.waypoints, nozzle_width_mm=2.0)
            assert 0.0 <= cov.coverage_fraction <= 1.0
            assert cov.uniformity_score >= 0.0
            assert cov.mean_distance_mm >= 0.0
