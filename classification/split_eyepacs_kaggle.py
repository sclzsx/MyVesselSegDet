from pathlib import Path
import numpy as np
import os
import pandas as pd
import random
import shutil
from PIL import Image
from tqdm import tqdm

def mkdir(d):
    if not os.path.exists(d): os.makedirs(d)

def split_5_classes():
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

def split_2_classes():
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

if __name__ == '__main__':
    # split_5_classes()
    split_2_classes()
