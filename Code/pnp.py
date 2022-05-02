from typing import List, Tuple
import numpy as np
from scipy.optimize import least_squares


def PnP(x: np.array, X: np.array, K: np.array) -> List[np.array]:

    print(x, x.shape)
    print(X, X.shape)
    x = np.column_stack((x, np.ones(len(x))))
    K_inv = np.linalg.inv(K)
    x_norm = np.dot(K_inv, x.transpose()).transpose()
    print(x_norm, x_norm.shape)

    A = np.zeros((0, 12))
    for i, pt in enumerate(X):
        pt = pt.reshape(1,4)
        Z = np.zeros((1,4))

        u, v, _ = x_norm[i]
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((pt, Z, Z)), 
                             np.hstack((Z, pt, Z)), 
                             np.hstack((Z, Z, pt))))
        a_i = u_cross.dot(X_tilde)

        A = np.append(A, a_i, axis=0)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3,4)
    R = P[:,:3]
    U, D, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)

    C  = P[:,3]
    t = np.dot(-R.transpose(), C).reshape(3,1)

    if(np.linalg.det(R) < 0):
        R = -R
        t = -t

    return R, t

def reprojection_error(x: np.array, X: np.array, P: np.array) -> float:
        x_ = np.dot(P, X)
        x_ = x_/x_[-1]

        return np.sum(np.subtract(x.reshape(2,1), x_[:2].reshape(2,1))**2)

def PnP_RANSAC(x: np.array, X: np.array, K: np.array, iterations: int=1000, epsilon: float=0.01) -> List[np.array]:

    best_R = None
    best_t = None
    best_inliers = None
    max_inliers = 0

    correspondences = np.arange(len(x)).tolist()
    for itr in range(iterations):
        inliers = list()
        correspondences_6 = np.random.choice(correspondences, 6, replace=False)
        R, t = PnP(x[correspondences_6], X[correspondences_6], K)

        I = np.identity(3)
        P = np.dot(K, np.dot(R, np.hstack((I, -t))))
        for i, pt in enumerate(x):
            err = reprojection_error(pt, X[i], P)
            if(err < epsilon):
                inliers.append(pt)

        if(len(inliers) > max_inliers):
            max_inliers = len(inliers)
            best_R = R
            best_t = t
            best_inliers = inliers

    print("Max Inliers:{}".format(max_inliers))