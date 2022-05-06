from typing import List, Tuple
import numpy as np
from scipy.optimize import least_squares

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

def linear_triangulation(r1: np.ndarray, c1: np.ndarray,
                         r2: np.ndarray, c2: np.ndarray,
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

# def get_triangulation_set(r_list: List[np.ndarray], c_list: List[np.ndarray],
#                           k: np.ndarray, matches: np.ndarray) -> List[np.ndarray]:
#     ref_r = np.identity(3)
#     ref_c = np.zeros((3, 1))
#     points1 = matches[:, 0:2]
#     points2 = matches[:, 2:4]
#     all_triangulated_points = []
#     for r, c in zip(r_list, c_list):
#         triangulated_points = linear_triangulation(c, r, ref_c, ref_r, k, points1, points2)
#         all_triangulated_points.append(triangulated_points)
#     return all_triangulated_points

# def disambiguate_camera_pose(r_list: List[np.ndarray], c_list: List[np.ndarray],
#                              all_triangulated_points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
#     max_index = -1
#     current_max = 0
#     for i, (r, c, triangulated_points) in enumerate(zip(r_list, c_list, all_triangulated_points)):
#         positive_points = 0
#         for point in triangulated_points:
#             depth = np.dot(r[2,:], (point.reshape(4,1)[:3]  - c.reshape(3, 1)))
#             if depth[0] > 0:
#                 positive_points += 1
#         if positive_points > current_max:
#             current_max = positive_points
#             max_index = i

#     triangulated_pts = all_triangulated_points[max_index]
#     triangulated_pts[:,0] = triangulated_pts[:,0]/triangulated_pts[:,-1]
#     triangulated_pts[:,1] = triangulated_pts[:,1]/triangulated_pts[:,-1]
#     triangulated_pts[:,2] = triangulated_pts[:,2]/triangulated_pts[:,-1]
#     triangulated_pts[:,3] = triangulated_pts[:,3]/triangulated_pts[:,-1]

#     return r_list[max_index], c_list[max_index].reshape(3,1), triangulated_pts

def get_all_triangulated_points(r_list: List[np.ndarray], c_list: List[np.ndarray],
                                k: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> List[np.ndarray]:
    r1 = np.identity(3)
    c1 = np.zeros((3, 1))
    all_triangulated_points = []
    for r2, c2 in zip(r_list, c_list):
        triangulated_points = linear_triangulation(r1, c1, r2, c2, k, points1, points2)
        all_triangulated_points.append(triangulated_points)
    return all_triangulated_points

def normalize_points(points: np.ndarray) -> np.ndarray:
    points[:,0] = points[:,0] / points[:,-1]
    points[:,1] = points[:,1] / points[:,-1]
    points[:,2] = points[:,2] / points[:,-1]
    points[:,3] = points[:,3] / points[:,-1]
    return points

def disambiguate_pose(r_list: List[np.ndarray], c_list: List[np.ndarray],
                      all_triangulated_points: List[np.ndarray]) -> int:
    idx = 0
    max_points = 0

    for i, (r, c, triangulated_pts) in enumerate(zip(r_list, c_list, all_triangulated_points)):
        c = c.reshape(-1, 1)
        r3 = r[2, :].reshape(1, -1)
        triangulated_pts = normalize_points(triangulated_pts)
        triangulated_pts = triangulated_pts[:, 0:3]
        points = positivity_constraint(triangulated_pts, r3, c)
        if points > max_points:
            idx = i
            max_points = points

    triangulated_pts = normalize_points(all_triangulated_points[idx])

    return r_list[idx], c_list[idx].reshape(3,1), triangulated_pts

def positivity_constraint(triangulated_points: np.ndarray, r3: np.ndarray, c: np.ndarray) -> int:
    points = 0
    for triangulated_point in triangulated_points:
        triangulated_point = triangulated_point.reshape(-1, 1)
        if r3.dot(triangulated_point - c) > 0 and triangulated_point[2] > 0:
            points += 1
    return points

def non_linear_triangulation(K, R0: np.array, C0: np.array, R1: np.array, C1: np.array, inliers: np.array, triangulated_pts: np.array) -> np.array:

    I = np.identity(3)
    P0 = np.dot(K, np.dot(R0, np.hstack((I, -C0))))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))

    def projection_error(P: np.array, x: np.array, X: np.array) -> float:
        x_ = np.dot(P, X)
        x_ = x_/x_[-1]

        # u = np.square(x[0] - np.divide(np.dot(P[0].transpose(), X), np.dot(P[2].transpose(), X)))
        # v = np.square(x[1] - np.divide(np.dot(P[1].transpose(), X), np.dot(P[2].transpose(), X)))
        # err = u + v

        return np.sum(np.subtract(x.reshape(2,1), x_[:2].reshape(2,1))**2)

    def projection_loss(X, x0, x1) -> float:
        loss = 0
        loss += projection_error(P0, x0, X)
        loss += projection_error(P1, x1, X)

        return loss

    def reprojection_loss(X) -> float:
        loss = 0
        for pt in range(len(inliers)):
            loss += projection_error(P0, inliers[pt,0], X[pt])
            loss += projection_error(P1, inliers[pt,1], X[pt])

        loss = loss/(len(inliers)*2)
        return loss

    print("Pre-optimization Loss: ", reprojection_loss(triangulated_pts))

    optimized_triangulated_pts = list()
    for pt in range(len(inliers)):
        X = least_squares(fun=projection_loss, x0=triangulated_pts[pt], args=[inliers[pt,0], inliers[pt,1]])
        optimized_triangulated_pts.append(X.x)

    print("Post-optimization Loss: ", reprojection_loss(optimized_triangulated_pts))

    optimized_triangulated_pts = np.array(optimized_triangulated_pts)
    return optimized_triangulated_pts