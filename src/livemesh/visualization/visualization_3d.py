"""3D visualization for conformal trajectory planning.

Provides matplotlib-based plotting functions for:
- Scaffold with void detection
- Honeycomb infill on curved surface
- Trajectory validation (desired vs actual)
- IK diagnostics (error, manipulability, joint limits)
- Hydrogel filling visualization

Translates the visualization sections from MuffinFresa_ConformalMapping.m
and visualize_hydrogel_filling.m.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Optional
from pathlib import Path


def plot_scaffold_void(
    vertices: np.ndarray,
    faces: np.ndarray,
    sharp_edges: np.ndarray,
    void_vid: np.ndarray,
    cyl_cy: float,
    cyl_cz: float,
    cyl_radius: float,
    save_path: Optional[str] = None,
):
    """Plot scaffold mesh with detected void boundary highlighted.

    Three panels: 3D view, UV (theta vs X) view, curvature check.
    """
    void_set = set(void_vid.tolist())
    void_edges = [e for e in sharp_edges if e[0] in void_set and e[1] in void_set]

    fig = plt.figure(figsize=(15, 5))

    # Panel 1: 3D scaffold with void
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, alpha=0.15, color='lightgray', edgecolor='gray', linewidth=0.1,
    )
    for e in void_edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Detected Void on Scaffold')
    ax1.view_init(25, 135)

    # Panel 2: UV projection (theta vs X_axial)
    ax2 = fig.add_subplot(132)
    for e in void_edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        t1 = np.degrees(np.arctan2(p1[1] - cyl_cy, p1[2] - cyl_cz))
        t2 = np.degrees(np.arctan2(p2[1] - cyl_cy, p2[2] - cyl_cz))
        ax2.plot([t1, t2], [p1[0], p2[0]], 'r-', linewidth=1.5)
    ax2.set_xlabel(r'$\theta$ (deg)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title(r'Void Boundary ($\theta$, X$_{axial}$)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Curvature check (theta vs Y)
    ax3 = fig.add_subplot(133)
    for e in void_edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        t1 = np.degrees(np.arctan2(p1[1] - cyl_cy, p1[2] - cyl_cz))
        t2 = np.degrees(np.arctan2(p2[1] - cyl_cy, p2[2] - cyl_cz))
        ax3.plot([t1, t2], [p1[1], p2[1]], 'r-', linewidth=1.5)
    theta_ref = np.linspace(-90, 90, 200)
    ax3.plot(theta_ref, cyl_cy + cyl_radius * np.sin(np.radians(theta_ref)), 'g--', linewidth=1.5)
    ax3.set_xlabel(r'$\theta$ (deg)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title(r'$\theta$ vs Y — green = R$\cdot$sin($\theta$)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_conformal_honeycomb(
    traj_uv: np.ndarray,
    traj_xyz: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    void_edges: list,
    save_path: Optional[str] = None,
):
    """Plot honeycomb infill on scaffold — deposition only (h <= 0)."""
    deposition_mask = traj_uv[2] <= 0
    dep_xyz = traj_xyz[:, deposition_mask]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, alpha=0.12, color='wheat', edgecolor='lightgray', linewidth=0.05,
    )

    for e in void_edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=1.5)

    ax.scatter(dep_xyz[0], dep_xyz[1], dep_xyz[2], s=0.5, c='blue', alpha=0.5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Conformal Honeycomb Infill')
    ax.view_init(25, 135)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_trajectory_validation(
    P_desired: np.ndarray,
    P_actual: np.ndarray,
    errors: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot desired vs actual TCP trajectories and tracking error.

    Args:
        P_desired: (3, N) desired positions (meters)
        P_actual: (3, N) actual positions from FK (meters)
        errors: (N,) position errors
    """
    fig = plt.figure(figsize=(14, 5))

    # 3D trajectory comparison
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(P_desired[0], P_desired[1], P_desired[2], 'b-', linewidth=1.2, label='Desired')
    ax1.plot(P_actual[0], P_actual[1], P_actual[2], 'r--', linewidth=0.8, label='Actual (IK)')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.legend()
    ax1.set_title('3D Trajectory')
    ax1.view_init(25, 135)

    # XY top view
    ax2 = fig.add_subplot(132)
    ax2.plot(P_desired[0], P_desired[1], 'b-', linewidth=1.2, label='Desired')
    ax2.plot(P_actual[0], P_actual[1], 'r--', linewidth=0.8, label='Actual')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Top View (XY)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Error plot
    ax3 = fig.add_subplot(133)
    ax3.plot(errors * 1000, 'm-', linewidth=1.0)
    ax3.axhline(0.5, color='k', linestyle='--', linewidth=0.8, label='0.5mm threshold')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Position Error [mm]')
    ax3.set_title('Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ik_diagnostics(
    errors: np.ndarray,
    mu_values: np.ndarray,
    phases: np.ndarray,
    q_solutions: np.ndarray,
    joint_limits: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot IK diagnostic information: error, manipulability, phases, joints.

    Args:
        errors: (N,) position errors
        mu_values: (N,) manipulability values
        phases: (N,) solving phase used (0=standard, 1-3=advanced)
        q_solutions: (8, N) joint solutions
        joint_limits: (8, 2) joint limits
    """
    N = len(errors)
    idx = np.arange(N)
    joint_names = ['prism_x', 'prism_y', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Error
    axes[0].plot(idx, errors * 1000, 'm-', linewidth=0.8)
    axes[0].axhline(0.5, color='k', linestyle='--', linewidth=0.7)
    axes[0].set_ylabel('Error [mm]')
    axes[0].set_title('Tracking Error Along Trajectory')
    axes[0].grid(True, alpha=0.3)

    # Manipulability
    axes[1].plot(idx, mu_values, 'b-', linewidth=0.8)
    axes[1].axhline(0.02, color='k', linestyle='--', linewidth=0.7, label=r'$\mu$ threshold')
    axes[1].set_ylabel(r'$\sqrt{\det(JJ^T)}$')
    axes[1].set_title('Manipulability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Phase activation
    colors = {1: 'green', 2: 'blue', 3: 'red'}
    for phase_val in [1, 2, 3]:
        mask = phases == phase_val
        if mask.any():
            axes[2].stem(idx[mask], np.full(mask.sum(), phase_val),
                        linefmt=colors[phase_val], markerfmt=f'{colors[phase_val][0]}.',
                        basefmt='', label=f'Phase {phase_val}')
    axes[2].set_xlabel('Point Index')
    axes[2].set_ylabel('Phase')
    axes[2].set_title('Solving Phase (0=standard)')
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Joint trajectories
    fig2, axes2 = plt.subplots(4, 2, figsize=(12, 10))
    for j in range(8):
        ax = axes2[j // 2, j % 2]
        ax.plot(idx, q_solutions[j], 'b-', linewidth=0.8)
        ax.axhline(joint_limits[j, 0], color='k', linestyle='--', linewidth=0.6)
        ax.axhline(joint_limits[j, 1], color='k', linestyle='--', linewidth=0.6)
        ax.set_title(f'{joint_names[j]} [{joint_limits[j,0]:.2f}, {joint_limits[j,1]:.2f}]')
        ax.grid(True, alpha=0.3)
        if j >= 6:
            ax.set_xlabel('Point Index')

    plt.suptitle('Joint Trajectories with Limits', fontsize=12)
    plt.tight_layout()
    if save_path:
        base = Path(save_path)
        plt.savefig(str(base.parent / f"{base.stem}_joints{base.suffix}"), dpi=150, bbox_inches='tight')
    plt.show()
