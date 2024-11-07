#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/23 16:09
# @Author  : yyywxk
# @File    : DI.py

import cv2
import numpy as np

def cal_distortion(src, dst, height, width, grid_h=12, grid_w=16):
    thr = 20
    num = src.shape[0] if src.shape[0] <= thr else thr
    rect_measure = []
    h_ = int(height / grid_h)
    w_ = int(width / grid_w)

    ceter_x, ceter_y = width / 2, height / 2

    for i in range(num):
        x, y = dst[i, 0, 0], dst[i, 0, 1]


        h0 = np.sqrt((ceter_x - x) ** 2 + (ceter_y - y) ** 2)

        ww, hh = src[i, 0, 0], src[i, 0, 1]
        h = np.sqrt((ceter_x - ww) ** 2 + (ceter_y - hh) ** 2)
        if h0 == 0.0:
            print(' 1 ')
            continue


        D_local = abs(h0 - h) / h0
        rect_measure.append(D_local)

    return np.mean(rect_measure)

# https://blog.csdn.net/weixin_45153969/article/details/131399378
# https://blog.csdn.net/hltt3838/article/details/129627346

img1 = cv2.imread('input.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread("gt.jpg", cv2.IMREAD_COLOR)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 暴力匹配器
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
ms = bf.match(des1, des2)
ms = sorted(ms, key=lambda x: x.distance)

# Draw
img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("matches.jpg", img3)

# 获取匹配点对的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)

rect_measure = cal_distortion(src_pts, dst_pts, img1.shape[0], img1.shape[1])
print('\nRectangling measure: ', rect_measure * 100, '%\n')
