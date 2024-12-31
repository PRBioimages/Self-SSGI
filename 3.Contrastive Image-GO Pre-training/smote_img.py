import argparse
import os
import random
import re
import torch.nn.functional as F
import numpy as np
import torch
from scipy.io import savemat
import scipy.io as sio
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def read_crop_img(img, crop_size, img_size):

    random_crop_size =crop_size
    x = int(np.random.uniform(0, img_size - random_crop_size))
    y = int(np.random.uniform(0, img_size - random_crop_size))
    crop_img = img[x:x + random_crop_size, y:y + random_crop_size]
    return crop_img


name_path = "/home/yjliang/PycharmProjects/CLIP/generate_data/divide_dataset/name.mat"
name = sio.loadmat(name_path)
name = name['name']



num=1
for i in range(len(name)):
    print(i)
    current_name = name[i][0][0]
    ##找到当前图片的文件夹，获得四通道的数据
    folder_path='./5407_4channel_data/'+current_name+'/'
    images_path = "./img_256/"
    if os.path.exists(folder_path):
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        current_images_path = images_path + str(num) + '.mat'


        random_index = random.choice(range(len(file_list)))

        current_random_img_path = folder_path + file_list[random_index]

        img=sio.loadmat(current_random_img_path)
        img =img['merged_image']


        # 定义数据增强的转换操作
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor()
        ])

        augmented_image = data_transform(img)
        augmented_image = augmented_image.permute(1, 2, 0)
        augmented_image=augmented_image.numpy()
        img_size=augmented_image.shape[0]
        augmented_image=read_crop_img(augmented_image,256, img_size)
        savemat(current_images_path, {'augmented_image': augmented_image})
        num=num+1



    else:
        continue

