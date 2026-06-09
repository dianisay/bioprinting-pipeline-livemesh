"""Inverse Kinematics: standard numerical IK + APF + Super-Twisting refinement.

Implements the multi-seed IK with Artificial Potential Field obstacle/limit
avoidance and Super-Twisting sliding-mode control for the 8-DOF system.

Translates Sections 8 and 7 (APF+STW) of test_obstacle_avoidance.m.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize

from .robot_model import (
    forward_kinematics_8dof,
    geometric_jacobian_8dof,
    manipulability,
    home_configuration,
    JOINT_LIMITS,
)


class IKParams:
    """Parameters for APF + Super-Twisting IK solver."""

    pos_tol: float = 1e-4           # 0.1 mm convergence
    err_threshold: float = 0.5e-3   # 0.5 mm — trigger advanced solving
    num_restarts: int = 15
    perturb_scale: float = 0.6      # null-space perturbation amplitude
    random_scale: float = 0.8       # full-space random amplitude
    max_iter: int = 200
    dt: float = 0.01
    alpha: float = 0.1              # APF repulsive weight
    eta: float = 0.001              # repulsive potential coefficient
    limit_margin: float = 0.12      # fraction of joint range for APF
    mu_threshold: float = 0.02      # manipulability threshold for DLS
    lambda_max: float = 0.15        # max DLS damping
    K1: float = 0.3                 # STW proportional gain
    K2: float = 0.1                 # STW integral gain
    v0: float = 1.0                 # max velocity
    kv: float = 4.0                 # velocity scaling
    e_u: float = 1e-6               # min norm for normalization
    omega_max: float = 0.5          # STW integrator saturation


def ik_cost(q: np.ndarray, T_target: np.ndarray, ori_weight: float = 1.0) -> float:
    """Cost function for scipy-based IK: position + orientation error."""
    T_fk = forward_kinematics_8dof(q)
    pos_err = T_target[:3, 3] - T_fk[:3, 3]
    R_err = T_target[:3, :3] @ T_fk[:3, :3].T
    # Rotation error via axis-angle
    trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    ori_err = np.arccos(trace)

    return np.sum(pos_err**2) + ori_weight * ori_err**2


def solve_ik_scipy(
    T_target: np.ndarray,
    q_init: np.ndarray,
    ori_weight: float = 1.0,
    max_iter: int = 500,
) -> Tuple[np.ndarray, float]:
    """Solve IK using scipy L-BFGS-B with joint limits.

    Returns:
        (q_solution, position_error)
    """
    bounds = [(JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1]) for i in range(8)]

    result = minimize(
        ik_cost,
        q_init,
        args=(T_target, ori_weight),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    q_sol = result.x
    T_fk = forward_kinematics_8dof(q_sol)
    pos_err = np.linalg.norm(T_target[:3, 3] - T_fk[:3, 3])

    return q_sol, pos_err


def apf_stw_refine(
    T_target: np.ndarray,
    q_init: np.ndarray,
    params: IKParams,
) -> Tuple[np.ndarray, Dict]:
    """APF + Super-Twisting sliding-mode IK refinement.

    Phase 3 from test_obstacle_avoidance.m: gradient-based refinement
    with joint limit repulsive potential and singularity-robust Jacobian.

    Returns:
        (q_solution, info_dict)
    """
    n = len(q_init)
    q = q_init.copy()
    Omega = np.zeros(n)
    q_dot_prev = np.zeros(n)
    jlim = JOINT_LIMITS

    p_des = T_target[:3, 3]
    R_des = T_target[:3, :3]

    best_q = q.copy()
    T_init = forward_kinematics_8dof(q)
    best_err = np.linalg.norm(p_des - T_init[:3, 3])
    stall_count = 0
    final_iter = 0

    for it in range(params.max_iter):
        T_curr = forward_kinematics_8dof(q)
        p_curr = T_curr[:3, 3]
        R_curr = T_curr[:3, :3]

        pos_err = p_des - p_curr
        R_err = R_des @ R_curr.T
        rot_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        dx = np.concatenate([pos_err, rot_err])

        pos_err_val = np.linalg.norm(pos_err)

        if pos_err_val < best_err - 1e-7:
            best_err = pos_err_val
            best_q = q.copy()
            stall_count = 0
        else:
            stall_count += 1

        final_iter = it + 1
        if pos_err_val < params.pos_tol:
            break
        if stall_count > 80:
            break

        # Jacobian and manipulability
        J = geometric_jacobian_8dof(q)
        JJt = J @ J.T
        mu_val = np.sqrt(max(0, np.linalg.det(JJt)))

        # Damped Least Squares (singularity-robust)
        if mu_val < params.mu_threshold:
            lam = params.lambda_max * (1 - (mu_val / params.mu_threshold) ** 2)
        else:
            lam = 0.0
        J_dls = J.T @ np.linalg.inv(JJt + lam**2 * np.eye(6))
        q_attract = J_dls @ dx

        # Repulsive potential from joint limits
        q_repel = np.zeros(n)
        for j in range(n):
            range_j = jlim[j, 1] - jlim[j, 0]
            if range_j < 1e-6:
                continue
            margin = params.limit_margin * range_j

            d_low = q[j] - jlim[j, 0]
            d_high = jlim[j, 1] - q[j]

            if d_low < margin and d_low > 0:
                q_repel[j] += params.eta / (d_low + 1e-6) ** 2
            if d_high < margin and d_high > 0:
                q_repel[j] -= params.eta / (d_high + 1e-6) ** 2

        # Combined gradient
        E_og = q_attract + params.alpha * q_repel

        # Velocity scaling
        d_goal = pos_err_val
        v_d = min(params.v0, params.kv * np.sqrt(d_goal))
        norm_E = np.linalg.norm(E_og)
        E_qd = v_d * E_og / max(norm_E, params.e_u)

        # Super-Twisting sliding mode
        S_q = q_dot_prev - E_qd
        abs_S = np.abs(S_q)
        sgn_S = np.sign(S_q + 1e-15)

        stw_prop = -params.K1 * np.sqrt(abs_S) * sgn_S
        Omega = Omega - params.K2 * sgn_S * params.dt
        Omega = np.clip(Omega, -params.omega_max, params.omega_max)

        q_dot = E_qd + stw_prop + Omega
        q_dot_prev = q_dot.copy()

        q = q + params.dt * q_dot
        q = np.clip(q, jlim[:, 0], jlim[:, 1])

    logger.debug(
        f"APF+STW refine: iterations={final_iter}, pos_err={best_err * 1000:.3f} mm"
    )
    return best_q, {"iter": final_iter, "pos_err": best_err}


class InverseKinematicsSolver:
    """Full IK solver with multi-seed + APF + STW.

    Implements the 3-phase solving strategy from test_obstacle_avoidance.m:
    - Phase 1: Standard IK with multiple weight sets
    - Phase 2: Null-space + random restarts
    - Phase 3: APF + Super-Twisting refinement
    """

    def __init__(self, params: Optional[IKParams] = None):
        self.params = params or IKParams()
        self.q_prev = home_configuration()
        self.q_prev_good = home_configuration()

    def solve(self, T_target: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Solve IK for a single target pose.

        Args:
            T_target: (4, 4) desired TCP pose

        Returns:
            (q_solution, info_dict)
        """
        params = self.params
        p_des = T_target[:3, 3]

        # Seeds
        q_home = home_configuration()
        seeds = [self.q_prev.copy(), self.q_prev_good.copy(), q_home.copy()]

        best_q = self.q_prev.copy()
        best_err = np.inf

        # Phase 1: Try seeds with different orientation weights
        ori_weights = [1.0, 0.3, 0.1, 0.01]
        for seed in seeds:
            q_sol, err = solve_ik_scipy(T_target, seed, ori_weight=1.0)
            if err < best_err:
                best_err = err
                best_q = q_sol
            if best_err < params.pos_tol:
                break

        if best_err > params.err_threshold:
            for w in ori_weights[1:]:
                q_sol, err = solve_ik_scipy(T_target, best_q, ori_weight=w)
                if err < best_err:
                    best_err = err
                    best_q = q_sol
                if best_err < params.pos_tol:
                    break

        phase = 0

        if best_err > params.err_threshold:
            phase = 1

            # Phase 2: Null-space perturbation
            J = geometric_jacobian_8dof(best_q)
            J_pinv = np.linalg.pinv(J)
            N_proj = np.eye(8) - J_pinv @ J

            for _ in range(5):
                delta = np.random.randn(8) * params.perturb_scale
                q_pert = best_q + N_proj @ delta
                q_pert = np.clip(q_pert, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

                q_sol, err = solve_ik_scipy(T_target, q_pert)
                if err < best_err:
                    best_err = err
                    best_q = q_sol
                    phase = 2
                if best_err < params.pos_tol:
                    break

            # Phase 2b: Random restarts
            if best_err > params.pos_tol:
                for _ in range(3):
                    q_rand = q_home + np.random.randn(8) * params.random_scale
                    q_rand = np.clip(q_rand, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                    q_sol, err = solve_ik_scipy(T_target, q_rand)
                    if err < best_err:
                        best_err = err
                        best_q = q_sol
                        phase = 2
                    if best_err < params.pos_tol:
                        break

            # Phase 3: APF + STW refinement
            if best_err > params.pos_tol:
                q_refined, info = apf_stw_refine(T_target, best_q, params)
                if info["pos_err"] < best_err:
                    best_err = info["pos_err"]
                    best_q = q_refined
                    phase = 3

        # Update chain
        self.q_prev = best_q.copy()
        if best_err < params.err_threshold:
            self.q_prev_good = best_q.copy()

        J_final = geometric_jacobian_8dof(best_q)
        mu = manipulability(J_final)

        logger.info(
            f"IK solve complete: phase={phase}, pos_err={best_err * 1000:.3f} mm, "
            f"manipulability={mu:.6f}"
        )

        return best_q, {
            "pos_err": best_err,
            "manipulability": mu,
            "phase": phase,
        }

    def solve_trajectory(self, trajectory_m: np.ndarray, R_targets: np.ndarray) -> Dict:
        """Solve IK for an entire trajectory.

        Args:
            trajectory_m: (3, N) TCP positions in meters
            R_targets: (3, 3, N) TCP orientations

        Returns:
            dict with q_solutions, errors, manipulability, phases
        """
        N = trajectory_m.shape[1]
        q_solutions = np.zeros((8, N))
        errors = np.zeros(N)
        mu_values = np.zeros(N)
        phases = np.zeros(N, dtype=int)

        self.q_prev = home_configuration()
        self.q_prev_good = home_configuration()

        logger.info(f"Solving IK trajectory: {N} points")

        for i in range(N):
            T_target = np.eye(4)
            T_target[:3, :3] = R_targets[:, :, i]
            T_target[:3, 3] = trajectory_m[:, i]

            q_sol, info = self.solve(T_target)

            q_solutions[:, i] = q_sol
            errors[i] = info["pos_err"]
            mu_values[i] = info["manipulability"]
            phases[i] = info["phase"]

            if (i + 1) % 100 == 0:
                logger.info(
                    f"IK progress [{i + 1}/{N}]: err={errors[i] * 1000:.3f} mm, "
                    f"mu={mu_values[i]:.4f}, phase={phases[i]}"
                )

        apf_count = int((phases > 0).sum())
        logger.info(
            f"IK trajectory complete: max_err={errors.max() * 1000:.3f} mm, "
            f"mean_err={errors.mean() * 1000:.3f} mm, "
            f"APF activated={apf_count}/{N} points"
        )

        return {
            "q_solutions": q_solutions,
            "errors": errors,
            "manipulability": mu_values,
            "phases": phases,
        }
