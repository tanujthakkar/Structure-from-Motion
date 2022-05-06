import argparse
import os
import numpy as np
import math
from matplotlib import pyplot as plt

from Code.image_utils import *
from Code.matching_utils import *
from Code.ransac_utils import *
from Code.essential_matrix import *
from Code.camera_utils import *
from Code.pnp import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str,
                        default='./Data/',
                        help='The path where the sfm data is stored.')
    parser.add_argument('-od', '--out_dir', type=str,
                        default='./Data/Outputs/',
                        help='The path where outputs are stored.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        default=False,
                        help='Should visualizations be saved?')

    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    visualize = args.visualize

    if not os.path.exists(data_dir):
        raise ValueError(f'The path {data_dir} does not exist!')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images = read_images(data_dir)
    all_matches = get_all_matches(data_dir)

    print("Matches")
    for i in all_matches:
        print(i.shape)
    all_inliers, f_mats = get_all_inliers(all_matches, out_dir)
    all_inliers = np.array(all_inliers, dtype=list)

    print("Inliers")
    for i in all_inliers:
        print(i.shape)
    f_mats = np.array(f_mats, dtype=list)

    K = camera_matrix()

    # print("F Mats")
    # for i, f in enumerate(f_mats):
    #     F, _ = cv2.findFundamentalMat(all_matches[i][:,0], all_matches[i][:,1], cv2.FM_LMEDS)
    #     print("CV2 F: ", F)
    #     print("F: ", f/f[-1,-1])
    # f_mats = np.array(f_mats, dtype=list)


    # E, _ = cv2.findEssentialMat(all_inliers[0][:,0], all_inliers[0][:,1], K, cv2.RANSAC)
    # print("E: ", E)

    # For first two images
    F = f_mats[0]
    print("\nF: ", F/F[-1,-1])
    E = get_essential_matrix(F)
    print("\nE: ", E)
    R, C = get_camera_poses(E)

    R0 = np.identity(3)
    C0 = np.zeros((3,1))

    all_triangulated_pts = get_all_triangulated_points(R, C, K, all_inliers[0][:,0], all_inliers[0][:,1])
    all_triangulated_pts = np.array(all_triangulated_pts)
    draw_triangulate_pts(all_triangulated_pts)

    R, C, triangulated_pts = disambiguate_pose(R, C, all_triangulated_pts)
    # draw_triangulate_pts([triangulated_pts])
    print("\nR: ", R)
    print("\nt: ", C)

    X = non_linear_triangulation(K, R0, C0, R, C, all_inliers[0], triangulated_pts)
    print(X.shape)
    draw_triangulate_pts([triangulated_pts, X])

    R_set = [R0, R]
    C_set = [C0, C]
    X_current = X
    X_set = X
    for img in range(2, len(images)):
        R_new, C_new, correspondences = PnP_RANSAC(img+1, images, all_inliers, X_current, K)
        R_opt, C_opt = non_linear_PnP(correspondences[0], correspondences[1], K, R_new, C_new)
        R_set.append(R_opt)
        C_set.append(C_opt)

        for i in range(img):
            pair = get_index(i+1, img+1)
            X_new = linear_triangulation(R_set[0], C_set[0], R_opt, C_opt, K, all_inliers[pair][:,0], all_inliers[pair][:,1])
            X_opt = non_linear_triangulation(K, R_set[0], C_set[0], R_opt, C_opt, all_inliers[pair], X_new)
            draw_triangulate_pts([X_opt, X_set])
            X_current = X_opt
            X_set = np.vstack((X_set, X_opt))
        input('q')

    if visualize:
        draw_all_matches(images, all_matches, out_dir)
        draw_all_inliers(images, all_matches, all_inliers, out_dir)
    #     draw_triangulate_pts(all_triangulated_pts)