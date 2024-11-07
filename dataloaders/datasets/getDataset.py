#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 18:31
# @Author  : yyywxk
# @File    : getDataset.py

import os
import random
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, slice_data=[0.4, 0.3, 0.3]):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: r"D:\Desktop\segmentation_2021\data"
    :param target_data_folder: 目标文件夹 r"D:\Desktop\segmentation_2021\a"
    :param slice_data: 划分数据比例比例  训练 验证 测试所占百分比
    :return:
    '''
    print("开始数据集划分")
    # class_names = os.listdir(src_data_folder)
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    split_names1 = ['mask', 'gt']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    class_name = 'input'
    current_class_data_path = os.path.join(src_data_folder, class_name)
    current_all_data = os.listdir(current_class_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
    val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
    test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
    train_stop_flag = current_data_length * slice_data[0]
    val_stop_flag = current_data_length * (slice_data[0] + slice_data[1])
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(current_class_data_path, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
            for name in split_names1:
                current_class_data_path1 = os.path.join(src_data_folder, name)
                train_folder1 = os.path.join(os.path.join(target_data_folder, 'train'), name)
                src_img_path1 = os.path.join(current_class_data_path1, current_all_data[i])
                copy2(src_img_path1, train_folder1)

        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_folder)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
            for name in split_names1:
                current_class_data_path1 = os.path.join(src_data_folder, name)
                val_folder1 = os.path.join(os.path.join(target_data_folder, 'val'), name)
                src_img_path1 = os.path.join(current_class_data_path1, current_all_data[i])
                copy2(src_img_path1, val_folder1)

        else:
            copy2(src_img_path, test_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1
            for name in split_names1:
                current_class_data_path1 = os.path.join(src_data_folder, name)
                test_folder1 = os.path.join(os.path.join(target_data_folder, 'test'), name)
                src_img_path1 = os.path.join(current_class_data_path1, current_all_data[i])
                copy2(src_img_path1, test_folder1)

        current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, slice_data[0], slice_data[1], slice_data[2],
                                                  current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))


if __name__ == '__main__':
    target_data_folder = "AIRD/"
    src_data_folder = "SRC/"
    data_set_split(src_data_folder, target_data_folder, slice_data=[0.6, 0.2, 0.2])
