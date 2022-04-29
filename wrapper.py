import argparse
import os

from Code.image_utils import draw_all_inliers, draw_all_matches, read_images
from Code.matching_utils import get_all_matches
from Code.ransac_utils import get_all_inliers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str,
                        default='./Data/',
                        help='The path where the sfm data is stored.')
    parser.add_argument('-od', '--out_dir', type=str,
                        default='./Data/Outputs/',
                        help='The path where outputs are stored.')
    parser.add_argument('-v', '--visualize', type=bool,
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
    all_inliers, f_mats = get_all_inliers(all_matches, out_dir)

    if visualize:
        draw_all_matches(images, all_matches, out_dir)
        draw_all_inliers(images, all_matches, all_inliers, out_dir)
