import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import random
import shutil
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from segmentation.models.LadderNet import LadderNet
from segmentation.lib.pre_processing import my_PreProc
import pandas as pd

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def split_5_classes(): # 把eyepacs_kaggle分为5类
    label = np.array(pd.read_csv('datasets/eyepacs_kaggle/trainLabels.csv'))
    print(label.shape)

    d = {0:[], 1:[], 2:[], 3:[], 4:[]}
    for i in range(label.shape[0]):
        name = label[i, 0]
        lab = label[i, 1]
        d[lab].append(name)

    for k, v in d.items():
        print(k, len(v))

    c0 = d[0]
    c1 = d[1]
    c2 = d[2]
    c3 = d[3]
    c4 = d[4]

    random.shuffle(c0)
    random.shuffle(c1)
    random.shuffle(c2)
    random.shuffle(c3)
    random.shuffle(c4)

    balance = True

    if balance:
        N = 10000000
        for i in range(5):
            if len(d[i]) < N:
                N = len(d[i])
        print(N)

    rate = 0.2

    L = [c0, c1, c2, c3, c4]
    for i in range(5):
        c = L[i]
        n = int(len(c) * rate)
        j = 0
        for name in tqdm(c):
            if balance and j >= N:
                break

            if j < n:
                DIR = 'datasets/eyepacs_multi_rgb/test/' + str(i)
            elif j >=n and j < n*2:
                DIR = 'datasets/eyepacs_multi_rgb/val/' + str(i)
            else:
                DIR = 'datasets/eyepacs_multi_rgb/train/' + str(i)

            mkdir(DIR)

            img = Image.open('datasets/eyepacs_kaggle/images/' + name + '.jpeg').convert('RGB')
            img = img.resize((512, 512), Image.ANTIALIAS)
            img.save(DIR + '/' + name + '.png')

            j += 1

def split_2_classes(): # 把eyepacs_kaggle分为2类
    label = np.array(pd.read_csv('datasets/eyepacs_kaggle/trainLabels.csv'))
    print(label.shape)

    N, P = [], []
    for i in range(label.shape[0]):
        name = label[i, 0]
        lab = label[i, 1]
        if lab > 0:
            N.append(name)
        else:
            P.append(name)

    random.shuffle(N)
    random.shuffle(P)
    print(len(N), len(P))

    lenN = min(len(N), len(P))
    P = P[:lenN]
    N = N[:lenN]

    rate = 0.2

    L = [P, N]
    for i in range(2):
        c = L[i]
        n = int(len(c) * rate)
        j = 0
        for name in tqdm(c):

            if j < n:
                DIR = 'datasets/eyepacs_binary_rgb/test/' + str(i)
            elif j >=n and j < n*2:
                DIR = 'datasets/eyepacs_binary_rgb/val/' + str(i)
            else:
                DIR = 'datasets/eyepacs_binary_rgb/train/' + str(i)

            mkdir(DIR)

            img = Image.open('datasets/eyepacs_kaggle/images/' + name + '.jpeg').convert('RGB')
            img = img.resize((512, 512), Image.ANTIALIAS)
            img.save(DIR + '/' + name + '.png')

            j += 1

def split_2_classes_small(): # 把eyepacs_kaggle分为2类
    label = np.array(pd.read_csv('datasets/eyepacs_kaggle/trainLabels.csv'))
    print(label.shape)

    N, P = [], []
    for i in range(label.shape[0]):
        name = label[i, 0]
        lab = label[i, 1]
        if lab > 0:
            N.append(name)
        else:
            P.append(name)

    random.shuffle(N)
    random.shuffle(P)
    print(len(N), len(P))

    lenN = min(len(N), len(P))
    P = P[:lenN]
    N = N[:lenN]

    rate = 0.2

    L = [P, N]
    for i in range(2):
        c = L[i]
        n = int(len(c) * rate)
        j = 0
        for name in tqdm(c):

            if j < n:
                DIR = 'datasets/eyepacs_binary_rgb/test/' + str(i)
            elif j >=n and j < n*2:
                DIR = 'datasets/eyepacs_binary_rgb/val/' + str(i)
            else:
                DIR = 'datasets/eyepacs_binary_rgb/train/' + str(i)

            mkdir(DIR)

            img = Image.open('datasets/eyepacs_kaggle/images/' + name + '.jpeg').convert('RGB')
            img = img.resize((512, 512), Image.ANTIALIAS)
            img.save(DIR + '/' + name + '.png')

            j += 1

def laddernet_seg_eye(img_path, save_path, unit_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    checkpoint = torch.load('results/segmentation/vessel_laddernet/best_model.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    data = Image.open(img_path).convert('RGB')
    rgb = np.array(data)
    data = rgb.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=0)
    data = my_PreProc(data)
    data = torch.from_numpy(data).float()

    with torch.no_grad():
        data = data.to(device)
        out_tensor = net(data)
    out = np.array(torch.max(out_tensor.data, 1)[1].squeeze().cpu())

    if unit_mode == 1:
        out = np.where(out > 0, 255, 0).astype('uint8')
        out = np.expand_dims(out, axis=-1)
        out = np.concatenate((rgb, out), axis=-1)
        out = Image.fromarray(out).convert('RGBA')
    elif unit_mode == 2:
        strength = 2.0
        out = out.astype('float')
        out = np.where(out > 0, strength, 0)
        out = np.expand_dims(out, axis=-1)
        out = np.concatenate((out, out, out), axis=-1)
        rgb = rgb.astype('float')
        denoised = cv2.GaussianBlur(rgb, (5, 5), 3)
        sharpen = rgb + (rgb - denoised) * out
        out = np.clip(sharpen, 0, 255).astype('uint8')
        out = Image.fromarray(out).convert('RGB')
    else:
        out = np.where(out > 0, 255, 0).astype('uint8')
        out = Image.fromarray(out).convert('RGB')

    out.save(save_path)

def segment_all_imgs(root, save_root_name, unit_mode):
    root_name = Path(root).name
    for img_path in Path(root).rglob('*.*'):
        img_path = str(img_path)
        save_path = img_path.replace(root_name, save_root_name).replace('jpeg', 'jpg')

        if os.path.exists(save_path):
            continue

        print(save_path)
        mkdir(str(Path(save_path).parent))

        laddernet_seg_eye(img_path, save_path, unit_mode=unit_mode)


if __name__ == '__main__':
    # 把eyepacs_kaggle分为2类,存到eyepacs_binary_rgb
    # split_2_classes()

    # 把eyepacs_binary_rgb分割锐化，作为分类网络的预处理
    segment_all_imgs('datasets/eyepacs_binary_rgb_tiny', 'eyepacs_binary_laddernet', unit_mode=0)

    segment_all_imgs('datasets/eyepacs_binary_rgb_tiny', 'eyepacs_binary_laddernet_rgba', unit_mode=1)

    segment_all_imgs('datasets/eyepacs_binary_rgb_tiny', 'eyepacs_binary_laddernet_sharpen', unit_mode=2)
