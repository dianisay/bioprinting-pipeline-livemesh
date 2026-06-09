"""STL surface analysis: load mesh, fit cylinder, detect void via sharp edges.

Translates Section 2 of MuffinFresa_ConformalMapping.m to Python.
Uses numpy-stl for mesh I/O and pure numpy for geometry.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


def load_stl(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load STL file and return vertices and triangle connectivity.

    Returns:
        vertices: (V, 3) unique vertex positions
        faces: (F, 3) triangle indices into vertices
    """
    from stl import mesh as stl_mesh

    stl = stl_mesh.Mesh.from_file(str(filepath))
    raw_verts = stl.vectors.reshape(-1, 3)

    # Deduplicate vertices
    vertices, inverse = np.unique(raw_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    return vertices, faces


def rotate_rx90(points: np.ndarray) -> np.ndarray:
    """Apply 90-degree rotation about X axis: Rx90 = [1 0 0; 0 0 -1; 0 1 0]."""
    Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    return points @ Rx90.T


def fit_cylinder_kasa(points: np.ndarray) -> Dict[str, float]:
    """Fit a circle in the YZ plane using the Kasa algebraic method.

    Assumes cylinder axis is along X.

    Returns:
        dict with keys: cy, cz, radius
    """
    y, z = points[:, 1], points[:, 2]
    A = np.column_stack([y, z, np.ones(len(y))])
    b = y**2 + z**2
    x_fit, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cy = x_fit[0] / 2.0
    cz = x_fit[1] / 2.0
    radius = np.sqrt(x_fit[2] + cy**2 + cz**2)

    return {"cy": cy, "cz": cz, "radius": radius}


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit normal for each triangle face.

    Returns:
        (F, 3) unit normals
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    normals = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15
    return normals / norms


def find_sharp_edges(
    vertices: np.ndarray, faces: np.ndarray, angle_threshold_deg: float = 35.0
) -> np.ndarray:
    """Find edges where dihedral angle exceeds threshold.

    Returns:
        (E, 2) array of vertex index pairs forming sharp edges
    """
    face_normals = compute_face_normals(vertices, faces)
    nf = len(faces)

    # Build edge-to-face map
    edges_all = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ])
    faces_all = np.tile(np.arange(nf), 3)
    edges_sorted = np.sort(edges_all, axis=1)

    # Group edges
    unique_edges, inverse = np.unique(edges_sorted, axis=0, return_inverse=True)

    sharp_edges = []
    for e_idx in range(len(unique_edges)):
        face_ids = faces_all[inverse == e_idx]
        if len(face_ids) == 2:
            n1 = face_normals[face_ids[0]]
            n2 = face_normals[face_ids[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            if angle_deg > angle_threshold_deg:
                sharp_edges.append(unique_edges[e_idx])

    return np.array(sharp_edges) if sharp_edges else np.empty((0, 2), dtype=int)


def connected_components(edges: np.ndarray) -> List[np.ndarray]:
    """Find connected components in an edge graph via BFS.

    Returns:
        List of arrays, each containing vertex indices of one component
    """
    if len(edges) == 0:
        return []

    all_verts = np.unique(edges.ravel())
    vert_to_idx = {v: i for i, v in enumerate(all_verts)}
    n = len(all_verts)

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for e in edges:
        a, b = vert_to_idx[e[0]], vert_to_idx[e[1]]
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        comp = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                    comp.append(nb)
        components.append(all_verts[comp])

    return components


def select_void_component(
    vertices: np.ndarray,
    components: List[np.ndarray],
    cyl_cy: float,
    cyl_cz: float,
    cyl_radius: float,
    edge_margin: float = 3.0,
) -> np.ndarray:
    """Score each component and select the one most likely to be the void.

    Scoring criteria (from MATLAB):
    - Not touching axial ends (+100)
    - Number of vertices (up to 80)
    - Closeness to cylinder radius (up to 20)
    - Span area in theta-axial space (up to 40)

    Returns:
        Array of vertex indices for the selected void component
    """
    x_all = vertices[:, 0]
    x_min_all, x_max_all = x_all.min(), x_all.max()

    best_score = -np.inf
    best_idx = -1

    for c_idx, vid in enumerate(components):
        pts = vertices[vid]
        ax_min, ax_max = pts[:, 0].min(), pts[:, 0].max()
        n_verts = len(pts)

        touches_end = (ax_min <= x_min_all + edge_margin) or (ax_max >= x_max_all - edge_margin)
        mean_rad_err = np.mean(np.abs(
            np.sqrt((pts[:, 1] - cyl_cy)**2 + (pts[:, 2] - cyl_cz)**2) - cyl_radius
        ))
        theta_r = np.arctan2(pts[:, 1] - cyl_cy, pts[:, 2] - cyl_cz)
        span_area = (theta_r.max() - theta_r.min()) * cyl_radius * (ax_max - ax_min)

        score = 0.0
        if not touches_end:
            score += 100
        score += min(n_verts, 80)
        score += max(0, 20 - mean_rad_err * 4)
        score += min(span_area / 40, 40)

        if score > best_score:
            best_score = score
            best_idx = c_idx

    if best_idx < 0:
        raise RuntimeError("No valid void component found")

    return components[best_idx]


def compute_void_bounds(
    vertices: np.ndarray,
    void_vid: np.ndarray,
    cyl_cy: float,
    cyl_cz: float,
    cyl_radius: float,
) -> Dict[str, float]:
    """Compute UV bounds and shell thickness of the detected void.

    Returns:
        dict with: theta_min, theta_max, x_min, x_max,
                   u_range (arc-length), v_range (axial),
                   void_width, void_length, shell_thickness
    """
    void_pts = vertices[void_vid]
    theta = np.arctan2(void_pts[:, 1] - cyl_cy, void_pts[:, 2] - cyl_cz)

    theta_min, theta_max = theta.min(), theta.max()
    x_min, x_max = void_pts[:, 0].min(), void_pts[:, 0].max()

    u_range = np.array([theta_min * cyl_radius, theta_max * cyl_radius])
    v_range = np.array([x_min, x_max])

    # Shell thickness from radial gap on one side
    side_band = (theta_max - theta_min) * 0.1
    left_side = np.abs(theta - theta_min) < side_band
    right_side = np.abs(theta - theta_max) < side_band

    if left_side.sum() >= 4:
        side_r = np.sqrt(
            (void_pts[left_side, 1] - cyl_cy)**2 + (void_pts[left_side, 2] - cyl_cz)**2
        )
    elif right_side.sum() >= 4:
        side_r = np.sqrt(
            (void_pts[right_side, 1] - cyl_cy)**2 + (void_pts[right_side, 2] - cyl_cz)**2
        )
    else:
        side_r = np.sqrt(
            (void_pts[:, 1] - cyl_cy)**2 + (void_pts[:, 2] - cyl_cz)**2
        )
    shell_thickness = side_r.max() - side_r.min()

    return {
        "theta_min": theta_min,
        "theta_max": theta_max,
        "x_min": x_min,
        "x_max": x_max,
        "u_range": u_range,
        "v_range": v_range,
        "void_width": float(np.diff(u_range)[0]),
        "void_length": float(np.diff(v_range)[0]),
        "shell_thickness": shell_thickness,
    }


def analyze_scaffold(stl_path: str, angle_threshold: float = 35.0) -> Dict:
    """Full pipeline: load STL → fit cylinder → detect void → compute bounds.

    Returns:
        dict with all scaffold geometry and void information
    """
    vertices, faces = load_stl(stl_path)
    vertices = rotate_rx90(vertices)

    cyl = fit_cylinder_kasa(vertices)

    sharp_edges = find_sharp_edges(vertices, faces, angle_threshold)
    components = connected_components(sharp_edges)

    void_vid = select_void_component(
        vertices, components, cyl["cy"], cyl["cz"], cyl["radius"]
    )

    bounds = compute_void_bounds(
        vertices, void_vid, cyl["cy"], cyl["cz"], cyl["radius"]
    )

    return {
        "vertices": vertices,
        "faces": faces,
        "cylinder": cyl,
        "sharp_edges": sharp_edges,
        "void_vid": void_vid,
        "void_bounds": bounds,
    }
