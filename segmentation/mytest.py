import os.path
import torch
from lib.extract_patches import *
import json
from lib.pre_processing import my_PreProc
from models.LadderNet import LadderNet
from PIL import Image
from pathlib import Path
import cv2
from metric import SegmentationMetric
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def cal_metric(net, image_path, label_path, mask_path):
    data = Image.open(image_path).convert('RGB')
    rgb = np.array(data)
    data = rgb.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=0)
    data = my_PreProc(data)
    _, _, h, w = data.shape
    hh = h // 16 * 16
    ww = w // 16 * 16
    data = data[:, :, :hh, :ww]

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = mask[:hh, :ww]
    mask = np.where(mask > 0, 1, 0)

    label = Image.open(label_path).convert('L')
    label = np.array(label)
    label = label[:hh, :ww]
    label = np.where(label > 0, 1, 0)
    label = label * mask

    input = data.squeeze()
    data = torch.from_numpy(data).float()
    with torch.no_grad():
        data = data.to(device)
        output = net(data)
    output = np.array(torch.max(output.data, 1)[1].squeeze().cpu())
    output = np.where(output > 0, 1, 0)
    output = output * mask

    labels = label.flatten()
    preds = output.flatten()
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    out_tensor = torch.from_numpy(output).long().unsqueeze(0).unsqueeze(0).to(device)
    label_tensor = torch.from_numpy(label).long().unsqueeze(0).unsqueeze(0).to(device)

    num_classes = 2
    metric = SegmentationMetric(num_classes)
    metric.update(out_tensor, label_tensor)
    pa, iou = metric.get()
    # print(pa, iou)
    # print(label.shape, input.shape, output.shape)
    show = np.hstack((label, input, output))
    show = (show * 255).astype('uint8')

    return acc, p, r, f1, pa, iou, show


def mytest(data_dir, save_root):
    if 'CHASEDB1' in data_dir:
        save_dir = save_root + '/show/CHASEDB1'
        mkdir(save_dir)
        save_path = save_root + '/CHASEDB1.json'
        label_paths = [i for i in Path(data_dir).rglob('*_1stHO.png')]
    elif 'DRIVE' in data_dir:
        save_dir = save_root + '/show/DRIVE'
        mkdir(save_dir)
        save_path = save_root + '/DRIVE.json'
        label_paths = [i for i in Path(data_dir).rglob('*_manual1.gif')]
    elif 'STARE' in data_dir:
        save_dir = save_root + '/show/STARE'
        mkdir(save_dir)
        save_path = save_root + '/STARE.json'
        label_paths = [i for i in Path(data_dir).rglob('*.ah.ppm')]
    else:
        return

    PA, IOU, ACC, P, R, F1 = [], [], [], [], [], []
    cnt = 0
    for label_path in label_paths:
        label_path = str(label_path)
        print(label_path)

        if 'CHASEDB1' in data_dir:
            image_path = label_path.replace('_1stHO', '').replace('png', 'jpg').replace('1st_label', 'images')
            mask_path = label_path.replace('_1stHO', '').replace('1st_label', 'mask')
        elif 'DRIVE' in data_dir:
            image_path = label_path.replace('_manual1.gif', '_test.tif').replace('1st_manual', 'images')
            mask_path = label_path.replace('_manual1.gif', '_test_mask.gif').replace('1st_manual', 'mask')
        elif 'STARE' in data_dir:
            image_path = label_path.replace('.ah.ppm', '.ppm').replace('png', 'jpg').replace('1st_labels_ah', 'images')
            mask_path = label_path.replace('im', 'mask_').replace('.ah.ppm', '.png').replace('1st_labels_ah', 'mask')
        else:
            return
        assert os.path.exists(image_path) == 1
        assert os.path.exists(label_path) == 1
        assert os.path.exists(mask_path) == 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
        checkpoint = torch.load('../results/segmentation/vessel_laddernet/best_model.pth')
        net.load_state_dict(checkpoint['net'])
        net.eval()

        acc, p, r, f1, pa, iou, show = cal_metric(net, image_path, label_path, mask_path)

        cv2.imwrite(save_dir + '/' + str(cnt) + '.jpg', show)

        ACC.append(acc)
        P.append(p)
        R.append(r)
        F1.append(f1)
        PA.append(pa)
        IOU.append(iou)

        cnt += 1

    metrics = {
        'accuracy': sum(ACC) / cnt,
        'precision': sum(P) / cnt,
        'recall': sum(R) / cnt,
        'f1_score': sum(F1) / cnt,
        'pa': sum(PA) / cnt,
        'iou': sum(IOU) / cnt,
    }
    for metric in metrics.items():
        print(metric)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    save_root = '../results/segmentationMetric'
    data_dirs = [
        '../datasets/vesselseg/CHASEDB1',
        '../datasets/vesselseg/DRIVE/test',
        '../datasets/vesselseg/STARE',
    ]
    for data_dir in data_dirs:
        mytest(data_dir, save_root)
