"""8-DOF Robot Model: XY Gantry (2 prismatic) + 6R Arm (myCobot).

Forward kinematics via DH parameters, Jacobian computation,
and manipulability analysis. All from scratch using numpy.

Translates the robot model from MuffinFresa_ConformalMapping.m and
the DH parameters from getFinalFrame.m.
"""

import numpy as np
from typing import Tuple, Optional


# myCobot DH parameters (modified DH convention)
# [alpha, a (mm), d (mm), theta_offset]
MYCOBOT_DH = np.array([
    [0.0,       0.0,   173.9,  0.0],       # J1
    [np.pi/2,   0.0,   0.0,    np.pi/2],   # J2 (offset +pi/2)
    [0.0,       135.0, 0.0,    0.0],        # J3
    [0.0,       120.0, 88.78,  np.pi/2],    # J4 (offset +pi/2)
    [np.pi/2,   0.0,   95.0,   0.0],        # J5
    [-np.pi/2,  0.0,   65.5,   0.0],        # J6
])

# Joint limits (radians) for full 8-DOF system
# [prism_x, prism_y, J1, J2, J3, J4, J5, J6]
JOINT_LIMITS = np.array([
    [-0.5, 0.5],     # prism_x (meters)
    [-0.5, 0.5],     # prism_y (meters)
    [-2.88, 2.88],   # J1 (~165 deg)
    [-2.88, 2.88],   # J2
    [-2.88, 2.88],   # J3
    [-2.88, 2.88],   # J4
    [-2.88, 2.88],   # J5
    [-2.88, 2.88],   # J6
])

JOINT_NAMES = ['prism_x', 'prism_y', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']


def dh_matrix(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Compute 4x4 homogeneous transformation from DH parameters.

    Uses the standard (Craig) DH convention.

    Args:
        alpha: link twist (rad)
        a: link length (mm)
        d: link offset (mm)
        theta: joint angle (rad)

    Returns:
        (4, 4) transformation matrix
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,    -st,    0,    a],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0,      0,      0,   1],
    ])


def forward_kinematics_6r(q6: np.ndarray) -> np.ndarray:
    """Compute FK for the 6R arm (myCobot) from joint angles.

    Args:
        q6: (6,) joint angles in radians

    Returns:
        (4, 4) end-effector pose relative to arm base
    """
    T = np.eye(4)
    for i in range(6):
        alpha, a, d, theta_off = MYCOBOT_DH[i]
        theta = q6[i] + theta_off
        T = T @ dh_matrix(alpha, a, d, theta)
    return T


def forward_kinematics_8dof(q: np.ndarray) -> np.ndarray:
    """Compute FK for full 8-DOF system (XY gantry + 6R arm).

    The gantry provides X,Y translation before the arm base.
    Gantry base is at a fixed position in world frame.

    Args:
        q: (8,) joint values [prism_x, prism_y, J1..J6]
            - prism_x, prism_y in meters
            - J1..J6 in radians

    Returns:
        (4, 4) TCP pose in world frame
    """
    px, py = q[0], q[1]

    # Gantry base transform (translation only)
    # Based on MATLAB: gantryWorldX=-0.5, gantryWorldY=-0.5, gantryWorldZ=-0.9
    T_gantry_base = np.eye(4)
    T_gantry_base[0, 3] = -0.5  # gantryWorldX
    T_gantry_base[1, 3] = -0.5  # gantryWorldY
    T_gantry_base[2, 3] = -0.9  # gantryWorldZ

    # Prismatic X
    T_px = np.eye(4)
    T_px[0, 3] = px

    # Prismatic Y
    T_py = np.eye(4)
    T_py[1, 3] = py

    # 6R arm FK (mm to m for the translation part)
    T_arm = forward_kinematics_6r(q[2:])
    T_arm[:3, 3] *= 0.001  # mm → m

    # Full chain
    T_world = T_gantry_base @ T_px @ T_py @ T_arm

    return T_world


def geometric_jacobian_8dof(q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Compute 6x8 geometric Jacobian via numerical differentiation.

    Uses central differences for accuracy.

    Args:
        q: (8,) joint configuration
        delta: perturbation size

    Returns:
        (6, 8) Jacobian [linear_velocity; angular_velocity]
    """
    J = np.zeros((6, 8))
    T0 = forward_kinematics_8dof(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]

    for i in range(8):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += delta
        q_minus[i] -= delta

        T_plus = forward_kinematics_8dof(q_plus)
        T_minus = forward_kinematics_8dof(q_minus)

        # Linear velocity (position derivative)
        J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * delta)

        # Angular velocity (rotation derivative via log of relative rotation)
        R_diff = T_plus[:3, :3] @ T_minus[:3, :3].T
        # Small angle approximation: omega ≈ (R_diff - I) vee / (2*delta)
        skew = (R_diff - R_diff.T) / 2.0
        J[3, i] = skew[2, 1] / (2 * delta)
        J[4, i] = skew[0, 2] / (2 * delta)
        J[5, i] = skew[1, 0] / (2 * delta)

    return J


def manipulability(J: np.ndarray) -> float:
    """Compute Yoshikawa manipulability index: sqrt(det(J*J^T)).

    Args:
        J: (6, n) Jacobian matrix

    Returns:
        Scalar manipulability value (0 at singularity)
    """
    JJt = J @ J.T
    det_val = np.linalg.det(JJt)
    return np.sqrt(max(0.0, det_val))


def check_joint_limits(q: np.ndarray, limits: np.ndarray = JOINT_LIMITS) -> Tuple[bool, np.ndarray]:
    """Check if joint configuration violates limits.

    Returns:
        (is_valid, violations) where violations[i] > 0 means violation amount
    """
    violations = np.zeros(len(q))
    violations = np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
    is_valid = np.all(violations < 1e-4)
    return is_valid, violations


def home_configuration() -> np.ndarray:
    """Return home (zero) configuration for the 8-DOF system."""
    return np.zeros(8)
