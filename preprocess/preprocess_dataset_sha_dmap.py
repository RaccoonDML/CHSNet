from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

import scipy
import scipy.spatial
import scipy.ndimage

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    # 快速排序的划分函数，找出第0,1,2,3近的四个点，第0个是自己
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def generate_adaptive_dmap_from_point(image, points):
    im_w, im_h = image.size
    dmap = np.zeros((im_w, im_h))
    if len(points)==0:
        return dmap.T
    else:
        for point in points:
            pt2d = np.zeros((im_w, im_h), dtype=np.float32)
            pt2d[int(point[0]), int(point[1])] += 1
            sigma = min(point[2], 15)
            tmp_dmap = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
            # rectify border error
            ct = np.sum(tmp_dmap)
            if abs(ct - 1) > 0.001:
                tmp_dmap *= (1 / ct)
            dmap += tmp_dmap
        return dmap.T


def generate_dmap_from_point(image, points):
    im_w, im_h = image.size
    dmap = np.zeros((im_w, im_h))
    for point in points:
        dmap[min(int(point[0]), im_w-1), min(int(point[1]), im_h-1)] += 1
    density_map = scipy.ndimage.filters.gaussian_filter(dmap, 15, mode='reflect')
    assert(abs(len(points)-np.sum(dmap)) < 1e-2)
    return density_map.T


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images','ground-truth').replace('IMG','GT_IMG').replace('.jpg', '.mat')
    points = loadmat(mat_path)
    points = points["image_info"][0,0][0,0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    # 过滤掉错误的标注
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(im, (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--origin-dir', default='./DATASET/ShanghaiTech/part_A',
                        help='original data directory')
    parser.add_argument('--data-dir', default='./DATASET/SHA-train-test-dmapfix15',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['train', 'test']:
        sub_dir = os.path.join(args.origin_dir, phase+'_data')
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

        im_list = glob(os.path.join(sub_dir, 'images', '*jpg'))
        for im_path in im_list:
            name = os.path.basename(im_path)
            print(phase + '-' + name)
            # 图像缩放，点标注提取和过滤
            im, points = generate_data(im_path)

            # 保存图像
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)

            # 生成并保存密度图
            dmap = generate_dmap_from_point(im, points)
            # or dmap = generate_adaptive_dmap_from_point(im, points)
            dmap_save_path = im_save_path.replace('.jpg', '_dmap.npy')
            np.save(dmap_save_path, dmap)

            # 保存点标注
            # if phase == 'train':  # for MAN BL-loss point annotation
            dis = find_dis(points)
            points = np.concatenate((points, dis), axis=1)  # N,2 -> N,3
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)
