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

    # For first two images
    K = camera_matrix()
    F = f_mats[0]
    print("\nF: ", F)
    E = get_essential_matrix(F)
    print("\nE: ", E)
    R, C = get_camera_poses(E)

    # all_triangulated_pts = get_triangulation_set(R, C, K, np.column_stack((all_inliers[0][:,0], all_inliers[0][:,1])))
    all_triangulated_pts = get_all_triangulated_points(R, C, K, all_inliers[0][:,0], all_inliers[0][:,1])
    all_triangulated_pts = np.array(all_triangulated_pts)
    draw_triangulate_pts(all_triangulated_pts)

    R, t, triangulated_pts = disambiguate_pose(R, C, all_triangulated_pts)
    draw_triangulate_pts([triangulated_pts])
    print("\nR: ", R)
    print("\nt: ", t)

    # optimized_triangulated_pts = non_linear_triangulation(K, R, t, all_inliers[0], triangulated_pts)
    # draw_triangulate_pts([triangulated_pts, optimized_triangulated_pts])

    # for img in range(2, len(images)):
    #     PnP_RANSAC(all_inliers[2][:,1], triangulated_pts, K)
    #     input('q')

    # print(all_inliers[0][:,1])
    # x = np.zeros(all_inliers[0].shape)
    # for pt in range(len(x)):
    #     x[pt,0] = all_inliers[0][pt,1]
    #     print(all_inliers[0][pt,1], x[pt,0])
    #     pair = get_index(2,3)
    #     print(pair)
    #     idx = np.where(all_inliers[pair][0] == x[pt,0])
    #     print(idx)
    #     input('q')

    if visualize:
        draw_all_matches(images, all_matches, out_dir)
        draw_all_inliers(images, all_matches, all_inliers, out_dir)
    #     draw_triangulate_pts(all_triangulated_pts)