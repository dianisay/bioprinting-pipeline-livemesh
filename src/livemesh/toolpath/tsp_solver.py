"""TSP solver for honeycomb cell visitation order optimization.

Uses Miller-Tucker-Zemlin (MTZ) formulation solved via PuLP (MILP).
Translates Section 3b of MuffinFresa_ConformalMapping.m.
"""

import logging

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def solve_tsp_mtz(distance_matrix: np.ndarray, time_limit: int = 60) -> np.ndarray:
    """Solve open-path TSP using MTZ subtour elimination via PuLP.

    Adds a dummy node with zero-cost arcs to convert the open-path TSP
    into a closed tour problem.

    Args:
        distance_matrix: (n, n) pairwise distances between cells
        time_limit: solver time limit in seconds

    Returns:
        (n,) permutation vector (0-indexed) giving optimal visitation order
    """
    import pulp

    n = len(distance_matrix)
    logger.info(f"TSP MTZ solve starting: {n} cells, time_limit={time_limit} s")
    N = n + 1  # Add dummy node for open path

    # Expand distance matrix with dummy node (zero cost)
    D = np.zeros((N, N))
    D[:n, :n] = distance_matrix

    # Create MILP problem
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

    # Decision variables: x[i][j] binary
    x = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

    # Subtour elimination variables: u[i] continuous
    u = {}
    for i in range(N):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=0, upBound=N - 1, cat=pulp.LpContinuous)

    # Objective: minimize total distance
    prob += pulp.lpSum(D[i, j] * x[i, j] for i in range(N) for j in range(N) if i != j)

    # Constraints: each node has exactly 1 outgoing arc
    for i in range(N):
        prob += pulp.lpSum(x[i, j] for j in range(N) if j != i) == 1

    # Constraints: each node has exactly 1 incoming arc
    for i in range(N):
        prob += pulp.lpSum(x[j, i] for j in range(N) if j != i) == 1

    # MTZ subtour elimination (real nodes only, not dummy)
    for i in range(n):
        for j in range(n):
            if i != j:
                prob += u[i] - u[j] + N * x[i, j] <= N - 1

    # Fix dummy node position
    prob += u[N - 1] == 0

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    if prob.status != pulp.constants.LpStatusOptimal:
        logger.warning(
            f"TSP solver status={pulp.LpStatus[prob.status]}, "
            f"using sequential order fallback"
        )
        return np.arange(n)

    # Extract tour by following arcs from dummy node
    X_sol = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and x[i, j].varValue is not None:
                X_sol[i, j] = round(x[i, j].varValue)

    tour_full = np.zeros(N, dtype=int)
    tour_full[0] = N - 1  # start at dummy
    for step in range(1, N):
        curr = tour_full[step - 1]
        nxt = np.where(X_sol[curr] > 0.5)[0]
        if len(nxt) == 0:
            break
        tour_full[step] = nxt[0]

    # Remove dummy node, keep only real nodes
    tour = tour_full[tour_full < n]
    logger.info(f"TSP MTZ solve complete: optimal tour over {len(tour)} cells")
    return tour


def compute_cell_centroids(grid: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
    """Compute UV centroids of cells from grid and index list.

    Args:
        grid: (ny, nx, 2) hexagonal grid positions
        cell_indices: (K, 2) array of (ix, iy) indices (1-based from MATLAB, 0-based here)

    Returns:
        (K, 2) UV centroids
    """
    centroids = np.zeros((len(cell_indices), 2))
    for i, (ix, iy) in enumerate(cell_indices):
        centroids[i] = grid[iy, ix]
    return centroids


def build_distance_matrix(centroids: np.ndarray, rise_penalty: float) -> np.ndarray:
    """Build TSP distance matrix with travel penalty.

    Distance = Euclidean(centroid_i, centroid_j) + 2*rise penalty.
    The rise penalty accounts for the nozzle having to lift before each hop.

    Args:
        centroids: (K, 2) cell centroids in UV space
        rise_penalty: 2 * rise altitude

    Returns:
        (K, K) distance matrix
    """
    from scipy.spatial.distance import cdist

    D = cdist(centroids, centroids) + rise_penalty
    np.fill_diagonal(D, 0)
    return D


def optimize_visitation_order(
    grid: np.ndarray,
    cell_indices: np.ndarray,
    rise: float = 20.0,
    time_limit: int = 60,
) -> np.ndarray:
    """Full TSP optimization pipeline for cell visitation.

    Args:
        grid: (ny, nx, 2) hexagonal grid
        cell_indices: (K, 2) cell indices (0-based)
        rise: travel altitude above surface (mm)
        time_limit: solver time limit

    Returns:
        Reordered cell_indices according to optimal tour
    """
    centroids = compute_cell_centroids(grid, cell_indices)
    D = build_distance_matrix(centroids, 2 * rise)

    tour_order = solve_tsp_mtz(D, time_limit)

    # Report savings
    n = len(cell_indices)
    seq_cost = sum(D[i, i + 1] for i in range(n - 1))
    opt_cost = sum(D[tour_order[i], tour_order[i + 1]] for i in range(n - 1))
    savings = 100 * (seq_cost - opt_cost) / (seq_cost + 1e-10)
    logger.info(
        f"TSP visitation order: sequential={seq_cost:.1f} mm, "
        f"optimal={opt_cost:.1f} mm, improvement={savings:.1f}%"
    )

    return cell_indices[tour_order]
