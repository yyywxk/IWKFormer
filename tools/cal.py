#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/5 16:58
# @Author  : yyywxk
# @File    : cal.py


import os
import numpy as np
import cv2
import torch
from metrics import calculate_psnr, calculate_ssim, calculate_fid_given_paths

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set the GPUs

import warnings

warnings.filterwarnings('ignore')

import lpips


def calc_lpips(img1_path, img2_path, net='alex', use_gpu=False):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.
    Returns
    -------
    dist01 : torch.Tensor
        学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

    References
    -------
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

    '''
    if (not os.path.exists(img1_path)) or (not os.path.exists(img2_path)):
        raise RuntimeError("no image found at '{}' or '{}'".format(img1_path, img2_path))

    assert (len(os.listdir(img1_path)) == len(os.listdir(img2_path)))

    loss_fn = lpips.LPIPS(net=net)
    if use_gpu:
        loss_fn.cuda()

    average_lpips_distance = 0
    for i, file in enumerate(os.listdir(img1_path)):
        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(img1_path, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(img2_path, file)))

            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()

            dist01 = loss_fn.forward(img0, img1)
            average_lpips_distance += dist01

        except Exception as e:
            print(e)

    return float(average_lpips_distance) / (i + 1)


class MyException(Exception):
    def __init__(self, msg):
        '''
        :param msg: Error
        '''
        self.msg = msg


base_dir = './BIQA_input/'
# input_dir = 'He_30x20/'
# input_dir = 'He_17x13/'
# input_dir = 'Nie_ours/'
# input_dir = 'Nies/'
input_dir = 'input/'

gt_dir = base_dir + 'gt/'
result_dir = base_dir + input_dir
if len(os.listdir(result_dir)) != len(os.listdir(gt_dir)):
    raise MyException(
        'The number of images is inconsistent in the directory: %s. Please check this directory! ' % result_dir)

print('PID Calculating...')
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
paths = []
paths.append(gt_dir)
paths.append(result_dir)
fid_value = calculate_fid_given_paths(paths, 1, device, 2048, 1)
print('LPIPS Calculating...')
lpips_value = calc_lpips(gt_dir, result_dir)

psnr_list = []
ssim_list = []

print('PSNR & SSIM\n')
for name_img in os.listdir(gt_dir):
    test_warp_image = cv2.imread(os.path.join(result_dir, name_img))
    test_warp_gt = cv2.imread(os.path.join(gt_dir, name_img))

    psnr_ = calculate_psnr(test_warp_image, test_warp_gt, input_order='HWC')
    ssim_ = calculate_ssim(test_warp_image, test_warp_gt, input_order='HWC')

    # psnr_ = skimage.measure.compare_psnr(test_warp_image, test_warp_gt, 255)
    # ssim_ = skimage.measure.compare_ssim(test_warp_image, test_warp_gt, data_range=255, multichannel=True)

    psnr_list.append(psnr_)
    ssim_list.append(ssim_)

psnr = np.mean(psnr_list)
ssim = np.mean(ssim_list)

print("===================Results Analysis After Saving ==================")
print('average psnr : {:.7f}'.format(psnr))
print('average ssim : {:.7f}'.format(ssim))
print('FID: ', fid_value)
print('LPIPS: ', lpips_value)
