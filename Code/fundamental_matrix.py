from typing import Tuple
import numpy as np

def normalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(points, axis=0)
    mean_x, mean_y = mean[0], mean[1]
    x_cap, y_cap = points[:, 0] - mean_x, points[:, 1] - mean_y
    s = (2 / np.mean(x_cap ** 2 + y_cap ** 2)) ** 0.5
    t_scale = np.diag([s, s, 1])
    t_trans = np.array([[1, 0, -mean_x], [0, 1, -mean_y], [0, 0, 1]])
    t = t_scale.dot(t_trans)
    points_norm = np.column_stack((points, np.ones(len(points))))
    points_norm = t.dot(points_norm.T).T
    return points_norm, t

def get_fundamental_matrix(points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
    assert points_1.shape == points_2.shape

    if points_1.shape[0] < 8:
        raise ValueError('Need at least 8 points to compute fundamental matrix!')

    points_1_norm, t1 = normalize(points_1)
    points_2_norm, t2 = normalize(points_2)

    a_matrix = []
    for point_1, point_2 in zip(points_1_norm, points_2_norm):
        x1, y1, _ = point_1
        x2, y2, _ = point_2
        a_matrix.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    a_matrix_np = np.array(a_matrix)
    _, _, v = np.linalg.svd(a_matrix_np, full_matrices=True)
    fundamental_matrix = v[-1, :]
    fundamental_matrix = fundamental_matrix.reshape(3, 3)

    # Force rank(fundamental_matrix) == 2
    u, s, vt = np.linalg.svd(fundamental_matrix)
    s = np.diag(s)
    s[2, 2] = 0
    fundamental_matrix = np.dot(u, np.dot(s, vt))

    # Normalize
    fundamental_matrix = np.dot(t2.T, np.dot(fundamental_matrix, t1))
    fundamental_matrix = fundamental_matrix / fundamental_matrix[2, 2]
    return fundamental_matrix
