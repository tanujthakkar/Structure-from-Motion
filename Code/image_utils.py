from typing import List
from cv2 import cv2
import numpy as np
import os

from Code.matching_utils import get_pair

def read_images(path: str) -> List[np.ndarray]:
    images = []
    file_names = sorted(os.listdir(path))
    for file_name in file_names:
        if file_name.endswith('jpg'):
            file_path = os.path.join(path, file_name)
            image = cv2.imread(file_path)
            images.append(image)
    return images

def draw_matches(image1: np.ndarray, image2: np.ndarray, matches: np.ndarray) -> np.ndarray:
    image1_copy = np.copy(image1)
    image2_copy = np.copy(image2)
    images_stacked = np.hstack((image1_copy, image2_copy))
    for match in matches:
        assert match.shape == (2, 2)
        point1 = (int(match[0, 0]), int(match[0, 1]))
        point2 = (int(match[1, 0]) + int(image1.shape[1]), int(match[1, 1]))
        images_stacked = cv2.circle(images_stacked, point1, 2, (0, 0, 255))
        images_stacked = cv2.circle(images_stacked, point2, 2, (0, 0, 255))
        images_stacked = cv2.line(images_stacked, point1, point2, (0, 0, 255), 1)
    return images_stacked

def find_in_matches(match: np.ndarray, matches: np.ndarray) -> bool:
    for match_i in matches:
        if match_i[0, 0] == match[0, 0] and match_i[1, 0] == match[1, 0] and \
           match_i[0, 1] == match[0, 1] and match_i[1, 1] == match[1, 1]:
            return True
    return False

def draw_inliers(image1: np.ndarray, image2: np.ndarray, matches: np.ndarray, inliers: np.ndarray) -> np.ndarray:
    image1_copy = np.copy(image1)
    image2_copy = np.copy(image2)
    images_stacked = np.hstack((image1_copy, image2_copy))
    for match in matches:
        if find_in_matches(match, inliers):
            continue
        point1 = (int(match[0, 0]), int(match[0, 1]))
        point2 = (int(match[1, 0]) + int(image1.shape[1]), int(match[1, 1]))
        images_stacked = cv2.circle(images_stacked, point1, 2, (0, 0, 255))
        images_stacked = cv2.circle(images_stacked, point2, 2, (0, 0, 255))
        images_stacked = cv2.line(images_stacked, point1, point2, (0, 0, 255), 1)
    for inlier in inliers:
        point1 = (int(inlier[0, 0]), int(inlier[0, 1]))
        point2 = (int(inlier[1, 0]) + int(image1.shape[1]), int(inlier[1, 1]))
        images_stacked = cv2.circle(images_stacked, point1, 2, (0, 255, 0))
        images_stacked = cv2.circle(images_stacked, point2, 2, (0, 255, 0))
        images_stacked = cv2.line(images_stacked, point1, point2, (0, 255, 0), 1)
    return images_stacked

def draw_all_matches(images: List[np.ndarray], all_matches: List[np.ndarray], path: str) -> None:
    for idx, matches in enumerate(all_matches):
        (i, j) = get_pair(idx)
        matched_image = draw_matches(images[i - 1], images[j - 1], matches)
        file_name = f'matched_{i}_{j}.jpg'
        matched_image_path = os.path.join(path, file_name)
        cv2.imwrite(matched_image_path, matched_image)

def draw_all_inliers(images: List[np.ndarray], all_matches: List[np.ndarray],
                     all_inliers: List[np.ndarray], path: str) -> None:
    for idx, (matches, inliers) in enumerate(zip(all_matches, all_inliers)):
        (i, j) = get_pair(idx)
        inliers_image = draw_inliers(images[i - 1], images[j - 1], matches, inliers)
        file_name = f'matched_inliers_{i}_{j}.jpg'
        inliers_image_path = os.path.join(path, file_name)
        cv2.imwrite(inliers_image_path, inliers_image)
    
