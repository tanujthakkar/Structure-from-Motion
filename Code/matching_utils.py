'''
1->2, 1->3, 1->4, 1->5, 1->6,
2->3, 2->4, 2->5, 2->6,
3->4, 3->5, 3->6,
4->5, 4->6,
5->6
'''

import os
import numpy as np
import cv2
from typing import Tuple, List

def get_index(i: int, j: int) -> int:
    if not i in range(1, 6) or not j in range(i + 1, 7) or j <= i:
        raise ValueError(f'The pair {(i, j)} is invalid for matching.')
    pairs = [(a, b) for a in range(1, 6) for b in range(a + 1, 7)]
    return pairs.index((i, j))

def get_pair(idx: int) -> Tuple[int, int]:
    if idx < 0 or idx >= 15:
        raise ValueError(f'The index {idx} is out of range for get_pair!')
    pairs = [(a, b) for a in range(1, 6) for b in range(a + 1, 7)]
    return pairs[idx]

def get_all_matches(path: str) -> List[np.ndarray]:
    all_matches = [[] for _ in range(15)]
    for file_name in sorted(os.listdir(path)):
        if not file_name.startswith('matching'):
            continue
        file_path = os.path.join(path, file_name)
        with open(file_path) as f:
            # get i from matching_i.txt
            i = int(file_name[-5])
            lines = f.readlines()
            assert len(lines) > 0
            num_features = int(lines[0].split(' ')[1])
            assert len(lines) == num_features + 1
            for line_idx in range(1, num_features + 1):
                line = lines[line_idx].strip()
                line = line.split(' ')
                num_matches = int(line[0])
                x1, y1 = float(line[4]), float(line[5])
                matches_in_line = [line[start:start+3] for start in range(6, len(line), 3)]
                assert len(matches_in_line) == num_matches - 1
                for match_in_line in matches_in_line:
                    assert len(match_in_line) == 3
                    j = int(match_in_line[0])
                    x2, y2 = float(match_in_line[1]), float(match_in_line[2])
                    all_matches[get_index(i, j)].append([[x1, y1], [x2, y2]])
                    # all_matches[get_index(i, j)].append([[y1, x1], [y2, x2]])
    all_matches_final = []
    for matches in all_matches:
        matches_np = np.array(matches)
        all_matches_final.append(matches_np)
    return all_matches_final

def get_matches(img0: np.array, img1: np.array, save: bool=False, visualize: bool=False) -> Tuple[np.array, np.array]:
        # Reference - https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

        print("\nEstimating feature pairs...")

        sift = cv2.SIFT_create()

        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(img0_gray, None)
        kp2, des2 = sift.detectAndCompute(img1_gray, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0,0] for i in range(len(matches))]

        x0 = np.empty([0,2])
        x1 = np.empty([0,2])

        for i,(m,n) in enumerate(matches):
            if m.distance < (0.5 * n.distance):
                x0 = np.append(x0, np.array([kp1[m.queryIdx].pt]), axis=0)
                x1 = np.append(x1, np.array([kp2[m.trainIdx].pt]), axis=0)
                matchesMask[i]=[1,0]

        print("Found {} feature pairs.".format(len(x0)))

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv2.DrawMatchesFlags_DEFAULT)

        img_matches = cv2.drawMatchesKnn(img0, kp1, img1, kp2, matches, None, **draw_params)

        if(visualize):
            # cv2.imshow("Inputs", np.hstack((img0, img1)))
            cv2.imshow("Matches", img_matches)
            cv2.waitKey()

        if(save):
            cv2.imwrite(os.path.join(save_path, dataset + '_matches.png'), img_matches)

        return x0, x1

def get_matches_set(image_set: np.array) -> np.array:

    for i, img in enumerate(image_set):
        for j, img_ in enumerate(image_set[i+1:]):
            x0, x1 = get_matches(img, img_, save=False, visualize=True)
            print(np.column_stack((x0, x1)).shape)