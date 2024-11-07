#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 10:10
# @Author  : yyywxk
# @File    : test.py

import argparse
import os
import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm
import time
import shutil

import torch

from dataloaders import make_data_loader
from metrics import calculate_psnr, calculate_ssim, calculate_fid_given_paths
from models.IWKFormer.model import RectanglingNetwork

from thop import clever_format, profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set the GPUs

import warnings

warnings.filterwarnings('ignore')

import lpips


def calculate_mask(result, input):
    total = result.shape[0] * result.shape[1]
    result_mask = total - np.sum(result > 127)
    input_mask = total - np.sum(input > 127)
    return 1 - result_mask / input_mask


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


def create_diff_image(image1_path, image2_path, output_path, name):
    """
    比较两张图像并将差异保存到文件中
    :param image1_path: 第一张图像的路径
    :param image2_path: 第二张图像的路径
    :param output_path: 输出图像的路径
    """
    with Image.open(image1_path) as image1, Image.open(image2_path) as image2:
        diff = ImageChops.difference(image1, image2)
        # diff.show()  # 显示差异图像
        diff.save(output_path+name)  # 保存差异图像到文件
        # print(name, 'has been computed successfully!\n')


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(description='PyTorch Codes for Deep Rectangling inference.')
    # dataset
    parser.add_argument('--dataset', type=str, default='AIRD',
                        choices=['DIR-D', 'AIRD'],
                        help='dataset name (default: DIR-D)')
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    # model path
    parser.add_argument('--model_path', type=str,
                        default='./run/AIRD/experiment_8/model_best.pth',
                        help='load your model')
    # define the image resolution
    parser.add_argument('--height', type=int, default=384, help='height of input images (default: 384)')
    parser.add_argument('--width', type=int, default=512, help='width of input images (default: 512)')
    # test settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing')
    parser.add_argument("--profile", dest='do_profiling', action='store_true', default=False,
                        help='Calculate amount of params and FLOPs. ')
    parser.add_argument("--cal", dest='direct_calculating', action='store_true', default=False,
                        help='After getting the predictive results, calculate the metrics directly . ')
    parser.add_argument("--dif", dest='compute_difference', action='store_true', default=False,
                        help='After getting the predictive results, calculate the difference directly . ')
    parser.add_argument('--resize', action='store_true', default=False, help='test using resized gt')

    parser.add_argument("--mask", action='store_true', default=False, help='generate mask.')

    # save path
    parser.add_argument('--save_path', type=str,
                        default='./test_results/',
                        help='save your prediction data')
    parser.add_argument('--save_mask_path', type=str,
                        default='./masks/',
                        help='save your prediction data')
    parser.add_argument('--save_dif_path', type=str,
                        default='./dif0/',
                        help='save difference data')

    # --------------------------------- Parameters of deeprect ----------------------------
    # define the mesh resolution
    parser.add_argument('--GRID_W', default=16, type=int, help='width of the mesh(default: 8)')
    parser.add_argument('--GRID_H', default=12, type=int, help='height of the mesh(default: 6)')

    # --------------------------------- Parameters of RIWKC ----------------------------
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--ite_num', type=int, default=2, help='iteration number')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            torch.cuda.empty_cache()
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.compute_difference:
        args.direct_calculating = True
        print('Directly compute the difference!\n')

    if args.direct_calculating:
        if os.path.exists(args.save_path):
            if os.path.getsize(args.save_path):
                print('The final results have been produced, calculate the metrics now!')
            else:
                raise FileNotFoundError('The testing directory %s is empty!' % args.save_path)
        else:
            raise FileNotFoundError('No such file or directory: %s' % args.save_path)
    else:
        if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
            print('Go predicting!\n')
            shutil.rmtree(args.save_path)
        if os.path.exists(args.save_mask_path) and args.mask:
            shutil.rmtree(args.save_mask_path)

        try:
            os.mkdir(args.save_path)
            if args.mask:
                os.mkdir(args.save_mask_path)
        except OSError:
            raise FileNotFoundError('The testing directory %s is empty!' % args.save_path)
        else:
            print("Successfully created the testing directory %s " % args.save_path)

    if (not os.path.isfile(args.model_path)) or (not os.path.exists(args.model_path)):
        raise RuntimeError("no checkpoint found at '{}'".format(args.model_path))

    return args


class Tester(object):
    def __init__(self, args):
        self.args = args
        # Load model
        model = RectanglingNetwork(args.n_colors, args.GRID_H, args.GRID_W, args.width, args.height, args.cuda,
                                   ite_num=args.ite_num)

        self.model = model

        pin_memory = False
        if args.cuda:
            pin_memory = True  # GPU will be faster
            self.model = self.model.cuda()
        checkpoint = torch.load(args.model_path)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print('Best PSNR=', checkpoint['best_pred'])
        print('Ep=', checkpoint['epoch'])

        # Weights calculation
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))

        # Define Dataloader
        self.test_loader, self.test_gt = make_data_loader(args, mode='test')

        # Calculate amount of params and FLOPs.
        if args.do_profiling:
            print('Calculate amount of params and FLOPs.')

            input = torch.randn(1, 3, args.height, args.width).cuda()
            mask = torch.randn(1, 3, args.height, args.width).cuda()
            macs, params = profile(model, inputs=(input, mask))
            macs, params = clever_format([macs, params], "%.3f")
            print("Params: {}, FLOPs: {}".format(params, macs))

            start = time.time()
            for i in range(20):
                with torch.no_grad():
                    input = torch.randn(10, 3, args.height, args.width).cuda()
                    mask = torch.randn(10, 3, args.height, args.width).cuda()
                    result = model(input, mask)
            end = time.time()
            print("FPS: {:.1f}".format(200 / (end - start)))
            exit(0)

    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Testing Images')
        psnr_list = []
        ssim_list = []
        if self.args.mask:
            mask_list = []
        for i, sample in enumerate(tbar):
            test_input, test_mask, test_gt, test_name = sample['image'], sample['mask'], sample['gt'], sample['name']
            if self.args.cuda:
                test_input, test_mask, test_gt = test_input.cuda(), test_mask.cuda(), test_gt.cuda()
            with torch.no_grad():
                test_warp_image_final, test_warp_mask_final = self.model(test_input, test_mask, train_flag=False)
                # test_warp_image_final = self.model(test_input, test_mask, train_flag=False)

            test_warp_image = (test_warp_image_final[0] + 1.0) * 127.5
            test_warp_gt = (test_gt[0] + 1.0) * 127.5
            test_warp_mask = (test_warp_mask_final[0] + 1.0) * 127.5
            # test_warp_input = (test_input[0] + 1.0) * 127.5
            test_mask = (test_mask[0][0:1] + 1.0) * 127.5

            if self.args.cuda:
                result = test_warp_image.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                gt = test_warp_gt.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                # result = test_warp_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
                if self.args.mask:
                    mask_result = test_warp_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
                    mask_input = test_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
            else:
                result = test_warp_image.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
                # result = test_warp_gt.permute(1, 2, 0).numpy().astype('uint8')
                gt = test_warp_gt.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
                if self.args.mask:
                    mask_result = test_warp_mask.permute(1, 2, 0).numpy().astype('uint8')
                    mask_input = test_mask.permute(1, 2, 0).numpy().astype('uint8')

            # if not self.args.resize:
            #     result = cv2.resize(result, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            psnr_ = calculate_psnr(result, gt, input_order='HWC')
            ssim_ = calculate_ssim(result, gt, input_order='HWC')
            if self.args.mask:
                mask_ = calculate_mask(mask_result, mask_input)
                # print('\nMask measure: ', str(i + 1).zfill(5), ".jpg: ", mask_ * 100, '%\n')
                print('\nMask measure: ', test_name[0], ' : ', mask_ * 100, '%\n')
                mask_list.append(mask_)

            psnr_list.append(psnr_)
            ssim_list.append(ssim_)

            # path = self.args.save_path + str(i + 1).zfill(5) + ".jpg"
            path = self.args.save_path + test_name[0]

            cv2.imwrite(path, result)
            if self.args.mask:
                # mask_path = self.args.save_mask_path + str(i + 1).zfill(5) + ".jpg"
                mask_path = self.args.save_mask_path + test_name[0]
                mask_result[mask_result > 127.5] = 255
                mask_result[mask_result < 127.5] = 0
                # mask_result = cv2.medianBlur(mask_result, 3)
                cv2.imwrite(mask_path, mask_result)

        psnr = np.mean(psnr_list)
        ssim = np.mean(ssim_list)
        if self.args.mask:
            mask_measure = np.mean(mask_list)

        # PID Score
        print('PID Calculating...')
        device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
        paths = []
        paths.append(self.test_gt)
        paths.append(self.args.save_path)
        fid_value = calculate_fid_given_paths(paths, 1, device, 2048, 1)
        print('LPIPS Calculating...')
        lpips_value = calc_lpips(self.test_gt, self.args.save_path)
        print("===================Results Analysis==================")
        print('average psnr : {:.7f}'.format(psnr))
        print('average ssim : {:.7f}'.format(ssim))
        print('FID: ', fid_value)
        print('LPIPS: ', lpips_value)
        if self.args.mask:
            print('mask measure: {:.7f} %'.format(mask_measure * 100))


class MyException(Exception):
    def __init__(self, msg):
        '''
        :param msg: Error
        '''
        self.msg = msg


def main(args):
    print('Start testing!')
    print(args)
    if not args.direct_calculating:
        tester = Tester(args)
        tester.test()

    from dataloaders import my_path
    if args.dataset == 'DIR-D':
        gt_dir = my_path(args.dataset) + 'testing/gt/'
    else:
        gt_dir = my_path(args.dataset) + 'test/gt/'
    result_dir = args.save_path
    if len(os.listdir(result_dir)) != len(os.listdir(gt_dir)):
        raise MyException(
            'The number of images is inconsistent in the directory: %s. Please check this directory! ' % result_dir)

    if args.compute_difference:
        print('Compute the difference!\n')
        if os.path.exists(args.save_dif_path):
            shutil.rmtree(args.save_dif_path)
        os.mkdir(args.save_dif_path)
        for name_img in tqdm(os.listdir(result_dir), desc='Calculating Images Difference'):
            test_image_path = os.path.join(result_dir, name_img)
            test_gt_path = os.path.join(gt_dir, name_img)
            create_diff_image(test_image_path, test_gt_path, args.save_dif_path, name_img)

        exit(0)

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
    for name_img in tqdm(os.listdir(gt_dir), desc='Calculating Metrics'):
        test_warp_image = cv2.imread(os.path.join(result_dir, name_img))
        test_warp_gt = cv2.imread(os.path.join(gt_dir, name_img))

        psnr_ = calculate_psnr(test_warp_image, test_warp_gt, input_order='HWC')
        ssim_ = calculate_ssim(test_warp_image, test_warp_gt, input_order='HWC')

        # psnr_ = skimage.measure.compare_psnr(test_warp_image, test_warp_gt, 255)
        # ssim_ = skimage.measure.compare_ssim(test_warp_image, test_warp_gt, data_range=255, multichannel=True)

        psnr_list.append(psnr_)
        ssim_list.append(ssim_)
        # print('\nPSNR: ', name_img, ' : ', psnr_)
        # print('\nSSIM: ', name_img, ' : ', ssim_)

    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)

    print("===================Results Analysis After Saving ==================")
    print('average psnr : {:.7f}'.format(psnr))
    print('average ssim : {:.7f}'.format(ssim))
    print('FID: ', fid_value)
    print('LPIPS: ', lpips_value)


if __name__ == "__main__":
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Finish testing!Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
