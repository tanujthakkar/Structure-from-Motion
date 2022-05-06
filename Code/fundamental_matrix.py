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

def normalize_image_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Expects points to be (-1, 2) shape numpy array.
    https://www5.cs.fau.de/fileadmin/lectures/2014s/Lecture.2014s.IMIP/exercises/4/exercise4.pdf
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry_2022.pdf
    '''
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError(f'Need an array of shape (-1, 2) to normalize.')

    t_x = np.mean(points[:, 0])
    t_y = np.mean(points[:, 1])
    x_translated = points[:, 0] - t_x
    y_translated = points[:, 1] - t_y
    scale = 2 / np.mean(np.sqrt(np.square(x_translated) + np.square(y_translated)))
    transform = np.array([
        [scale, 0, -t_x * scale],
        [0, scale, -t_y * scale],
        [0, 0, 1]
    ])
    points_normalized = np.column_stack((points, np.ones(points.shape[0])))
    points_normalized = np.dot(transform, points_normalized.T).T
    return points_normalized, transform

def get_fundamental_matrix(points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
    assert points_1.shape == points_2.shape

    if points_1.shape[0] < 8:
        raise ValueError('Need at least 8 points to compute fundamental matrix!')

    points_1_norm, t1 = normalize_image_points(points_1)
    points_2_norm, t2 = normalize_image_points(points_2)

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
    # fundamental_matrix = fundamental_matrix / fundamental_matrix[-1, -1]
    return fundamental_matrix

# def get_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
#     error_message = ''
#     if len(points1.shape) != 2 or points1.shape[1] != 2:
#         error_message = f'The points1 are not of the correct shape. Need: (-1, 2), Got: {points1.shape}'
#     elif len(points2.shape) != 2 or points2.shape[1] != 2:
#         error_message = f'The points2 are not of the correct shape. Need: (-1, 2), Got: {points2.shape}'
#     elif points1.shape[0] != points2.shape[0]:
#         error_message = f'Need equal number of points to compute fundamental matrix.'
#     elif points1.shape[0] < 8:
#         error_message = f'Need at least 8 points to compute fundamental matrix.'

#     if len(error_message) > 0:
#         raise ValueError(error_message)

#     points1_normalized, transform1 = normalize_image_points(points1)
#     points2_normalized, transform2 = normalize_image_points(points2)

#     a_matrix_np = []
#     for (u1, v1, _), (u2, v2, _) in zip(points1_normalized, points2_normalized):
#         a_matrix_np.append([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1])
#     a_matrix_np = np.array(a_matrix_np)
#     _, _, vt = np.linalg.svd(a_matrix_np, full_matrices=True)
#     fundamental_matrix = vt[-1, :]
#     fundamental_matrix = fundamental_matrix.reshape(3, 3)

#     u, s, vt = np.linalg.svd(fundamental_matrix)
#     s = np.diag(s)
#     s[2, 2] = 0
#     fundamental_matrix = np.dot(u, np.dot(s, vt))

#     fundamental_matrix = np.dot(transform1.T, np.dot(fundamental_matrix, transform2))

#     # fundamental_matrix = fundamental_matrix / fundamental_matrix[-1, -1]
#     return fundamental_matrix 
