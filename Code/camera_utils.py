from typing import List, Tuple
import numpy as np

def get_camera_poses(e_mat: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    u, _, vt = np.linalg.svd(e_mat)
    w = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])
    r_list = []
    c_list = []
    r_list.append(np.dot(u, np.dot(w, vt)))
    r_list.append(np.dot(u, np.dot(w, vt)))
    r_list.append(np.dot(u, np.dot(w.T, vt)))
    r_list.append(np.dot(u, np.dot(w.T, vt)))
    c_list.append(u[:, 2])
    c_list.append(-u[:, 2])
    c_list.append(u[:, 2])
    c_list.append(-u[:, 2])

    # Correct poses if det(R) < 0
    for i in range(4):
        if np.linalg.det(r_list[i]) < 0:
            r_list[i] = -r_list[i]
            c_list[i] = -c_list[i]

    return r_list, c_list

def linear_triangulation(c1: np.ndarray, r1: np.ndarray,
                         c2: np.ndarray, r2: np.ndarray,
                         k: np.ndarray, points1: np.ndarray,
                         points2: np.ndarray) -> np.ndarray:
    eye = np.identity(3)
    c1 = c1.reshape(3, 1)
    c2 = c2.reshape(3, 1)
    projection_matrix1 = np.dot(k, np.dot(r1, np.hstack((eye, -c1))))
    projection_matrix2 = np.dot(k, np.dot(r2, np.hstack((eye, -c2))))
    p1_1 = projection_matrix1[0, :].reshape(1, 4)
    p1_2 = projection_matrix1[1, :].reshape(1, 4)
    p1_3 = projection_matrix1[2, :].reshape(1, 4)
    p2_1 = projection_matrix2[0, :].reshape(1, 4)
    p2_2 = projection_matrix2[1, :].reshape(1, 4)
    p2_3 = projection_matrix2[2, :].reshape(1, 4)
    triangulated_points = []

    for point1, point2 in zip(points1, points2):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        a_mat = []
        a_mat.append((y1 * p1_3) - p1_2)
        a_mat.append(p1_1 - (x1 * p1_3))
        a_mat.append((y2 * p2_3) - p2_2)
        a_mat.append(p2_1 - (x2 * p2_3))
        a_mat_np = np.array(a_mat).reshape(4, 4)
        _, _, vt = np.linalg.svd(a_mat_np)
        triangulated_points.append(vt[-1])

    triangulated_points_np = np.array(triangulated_points)
    return triangulated_points_np
