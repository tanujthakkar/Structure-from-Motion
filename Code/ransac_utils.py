from typing import Tuple, List
import numpy as np
import os
from tqdm import tqdm

from Code.fundamental_matrix import get_fundamental_matrix
from Code.matching_utils import get_pair

def saved(path: str) -> bool:
    inliers_file_names = [file_name for file_name in os.listdir(path) if file_name.startswith('inliers')]
    f_mats_file_names = [file_name for file_name in os.listdir(path) if file_name.startswith('f_mat')]
    if len(inliers_file_names) == 15 and len(f_mats_file_names) == 15:
        return True
    else:
        return False

def save(all_inliers: List[np.ndarray], f_mats: List[np.ndarray], path: str) -> None:
    for idx, inliers in enumerate(all_inliers):
        (i, j) = get_pair(idx)
        inliers_file_name = os.path.join(path, f'inliers_{i}_{j}.npy')
        np.save(inliers_file_name, inliers)
    for idx, f_mat in enumerate(f_mats):
        (i, j) = get_pair(idx)
        f_mat_file_name = os.path.join(path, f'f_mat_{i}_{j}.npy')
        np.save(f_mat_file_name, f_mat)

def read_saved(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    all_inliers = []
    inliers_file_names = sorted([file_name for file_name in os.listdir(path) if file_name.startswith('inliers')])
    for inliers_file_name in inliers_file_names:
        inliers_file_path = os.path.join(path, inliers_file_name)
        inliers = np.load(inliers_file_path)
        all_inliers.append(inliers)
    f_mats = []
    f_mats_file_names = sorted([file_name for file_name in os.listdir(path) if file_name.startswith('f_mat')])
    for f_mat_file_name in f_mats_file_names:
        f_mat_file_path = os.path.join(path, f_mat_file_name)
        f_mat = np.load(f_mat_file_path)
        f_mats.append(f_mat)
    return all_inliers, f_mats

def error_func(pts1, pts2, F): 
    x1, x2 = pts1, pts2
    x1tmp = np.array([x1[0], x1[1], 1])
    x2tmp = np.array([x2[0], x2[1], 1]).T
    error = np.dot(x2tmp, np.dot(F, x1tmp))
    return np.abs(error)

def get_inliers_ransac(matches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(1)
    iterations = 1000
    threshold = 0.001
    best_inliers_len = 0
    best_inliers = np.array([])
    best_f_mat = np.array([])
    if matches.shape[0] <= 0:
        return best_inliers, best_f_mat
    for _ in tqdm(range(iterations)):
        random_m_idx = np.random.choice(matches.shape[0], size=8)
        random_matches = matches[random_m_idx]
        points1 = random_matches[:, 0, :]
        points2 = random_matches[:, 1, :]
        f_mat = get_fundamental_matrix(points1, points2)
        inliers = []
        for match in matches:
            error = error_func(match[0], match[1], f_mat)
            if error < threshold:
                inliers.append(match)
        inliers_np = np.array(inliers)
        if len(inliers) > best_inliers_len:
            best_inliers_len = len(inliers)
            best_inliers = inliers_np
            best_f_mat = f_mat
    return best_inliers, best_f_mat

def get_all_inliers(all_matches: List[np.ndarray], path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print("\nEstimating inliers using RANSAC...")

    if saved(path):
        return read_saved(path)
    all_inliers = []
    f_mats = []
    for matches in all_matches:
        inliers, f_mat = get_inliers_ransac(matches)
        all_inliers.append(inliers)
        f_mats.append(f_mat)
    save(all_inliers, f_mats, path)
    return all_inliers, f_mats
