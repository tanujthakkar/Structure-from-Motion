import numpy as np

def camera_matrix() -> np.ndarray:
    return np.array([
        [568.996140852, 0, 643.21055941],
        [0, 568.988362396, 477.982801038],
        [0, 0, 1]
    ])

def get_essential_matrix(f_mat: np.ndarray) -> np.ndarray:
    k = camera_matrix()
    e_mat = k.T.dot(f_mat).dot(k)
    u, _, v = np.linalg.svd(e_mat)
    s = np.diag([1, 1, 0])
    e_mat = np.dot(u, np.dot(s, v))
    return e_mat