from LadderNet import LadderNet
import torch
from PIL import Image
import numpy as np
from pre_processing import my_PreProc
import cv2
from pathlib import Path


def load_seg_model(path):
    net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).cpu()
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net.cpu()
    return net


def do_seg(pt_path, img_path):
    seg_model = load_seg_model(pt_path)

    data = Image.open(img_path).convert('RGB')
    rgb = np.array(data)
    data = rgb.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=0)
    data = my_PreProc(data)
    pre_process_np = data

    pre_process_tensor = torch.from_numpy(pre_process_np).float()
    with torch.no_grad():
        pre_process_tensor = pre_process_tensor.cpu()
        seg_out_tensor = seg_model(pre_process_tensor)
    seg_out = np.array(torch.max(seg_out_tensor.data, 1)[1].squeeze().cpu())
    seg_out = np.where(seg_out > 0, 255, 0).astype('uint8')
    seg_out = cv2.cvtColor(seg_out, cv2.COLOR_GRAY2BGR)

    return seg_out


if __name__ == '__main__':
    pt_path = 'weights/segmentation.pth'

    for image_path in Path('images').glob('*.png'):
        image_path = str(image_path)

        seg_out = do_seg(pt_path, image_path)

        cv2.imshow('seg_out', seg_out)
        cv2.waitKey()
