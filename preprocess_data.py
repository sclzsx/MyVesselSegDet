import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import random
import shutil
import torch
from Nets.UNet import UNet
from torchvision import transforms
from matplotlib import pyplot as plt


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def cv2_preprocess_eye(img_path, lab_path, do_crop=1, dst_size=(512, 512)):
    bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    mask = cv2.resize(gray, (W // 16, H // 16))  # 小图，降噪，加速
    mask = cv2.GaussianBlur(mask, (3, 3), 2)  # 平滑
    mask = mask.astype('float32') / 255  # 归一化
    mask = mask ** 2  # 暗区压暗
    mask = cv2.resize(mask, (W, H))  # 复原尺寸
    mask = np.where(mask > 0.02, 1, 0).astype('uint8')  # 滤噪，二值化

    if do_crop:
        x, y, w, h = cv2.boundingRect(mask.astype('uint8') * 255)  # 外接矩
        m = 10
        h0 = max(y - m, 0)
        w0 = max(x - m, 0)
        h1 = min(y + h + m, H)
        w1 = min(x + w + m, W)
    else:
        h0, h1, w0, w1 = 0, H, 0, W

    gray = gray[h0:h1, w0:w1]
    mask = mask[h0:h1, w0:w1]
    bgr = bgr[h0:h1, w0:w1]
    # print(gray.shape, mask.shape)

    img = cv2.GaussianBlur(gray, (3, 3), 1)  # 降噪
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 对比度增强
    img = clahe.apply(img)
    img = img.astype('float32') / 255  # 归一化
    img = img ** (1 / 2.2)  # 暗区提亮

    img2 = cv2.GaussianBlur(img, (3, 3), 3)
    img = img + (img - img2) * 2  # 锐化

    lab = None

    if dst_size is not None:
        img = cv2.resize(img, dst_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dst_size, interpolation=cv2.INTER_LINEAR)
        bgr = cv2.resize(bgr, dst_size, interpolation=cv2.INTER_CUBIC)

        if lab_path is not None:
            lab = Image.open(lab_path).convert('L')
            lab = np.array(lab)
            lab = lab[h0:h1, w0:w1]
            lab = cv2.resize(lab, dst_size, interpolation=cv2.INTER_LINEAR)
            lab = np.where(lab > 0, 1, 0)
            lab = (lab * mask).astype('uint8')

    img = (np.clip(img * mask, 0, 1) * 255).astype('uint8')  # 截断
    mask = mask.astype('bool')

    return img, lab, mask, bgr


def print_np_info(lab, name):
    print(name, lab.shape, lab.dtype, np.min(lab), np.max(lab), np.mean(lab))


def split_seg_dataset(root, save_root):
    mkdir(save_root)
    for img_path in Path(root).rglob('*.*'):
        img_name = img_path.name
        img_parent = img_path.parent.name
        if img_parent != 'images':
            continue

        img_path = str(img_path)
        if 'CHASEDB1' in img_path:
            lab_path = img_path.replace('images', '1st_label').replace('.jpg', '_1stHO.png')
        elif 'STARE' in img_path:
            lab_path = img_path.replace('images', '1st_labels_ah').replace('.ppm', '.ah.ppm')
        elif 'DRIVE' in img_path:
            lab_path = img_path.replace('images', '1st_manual').replace('_test.tif', '_manual1.gif').replace(
                '_training.tif', '_manual1.gif')
        else:
            continue

        # print(img_path, lab_path)
        assert (os.path.exists(lab_path))

        img, lab, mask, bgr = cv2_preprocess_eye(img_path, lab_path)
        print_np_info(img, 'img')
        print_np_info(mask, 'mask')
        print_np_info(lab, 'lab')

        img = Image.fromarray(img)
        img.save(save_root + '/' + img_name + '.png')
        lab = Image.fromarray(lab)
        lab.save(save_root + '/' + img_name + '_lab.png')


def aug_seg_data(data_root, save_root, aug_rate=30):
    mkdir(save_root)
    for i, lab_path in enumerate(Path(data_root).glob('*_lab.png')):
        lab_path = str(lab_path)
        img_path = lab_path.replace('_lab', '')

        img = Image.open(img_path)
        lab = Image.open(lab_path)

        for j in range(aug_rate):
            hflip_flag = random.random() < 0.5
            vflip_flag = random.random() < 0.5
            # rot_angle = random.randint(0, 360)
            rot_angle = random.choice([0, 90, 270])
            if hflip_flag:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip_flag:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                lab = lab.transpose(Image.FLIP_TOP_BOTTOM)
            if rot_angle:
                img = img.rotate(rot_angle, expand=False)
                lab = lab.rotate(rot_angle, expand=False)

            name = str(i) + '_' + str(j)
            print('save_name:{}, hflip_flag:{}, vflip_flag:{}, rot_angle:{}'.format(name, hflip_flag, vflip_flag,
                                                                                    rot_angle))
            img.save(save_root + '/' + name + '.png')
            lab.save(save_root + '/' + name + '_lab.png')


def split_aug_data(data_root, save_root, test_rate=0.2):
    save_dir_trainval = save_root + '/trainval'
    save_dir_test = save_root + '/test'
    mkdir(save_dir_trainval)
    mkdir(save_dir_test)
    lab_paths = [lab_path for lab_path in Path(data_root).glob('*_lab.png')]
    num = len(lab_paths)
    test_num = int(num * test_rate)
    random.shuffle(lab_paths)
    for i, lab_path in enumerate(lab_paths):
        img_name = lab_path.name.split('_lab')[0]
        img_path = str(lab_path).replace('_lab', '')
        print(i, img_path)
        if i < test_num:
            os.rename(str(lab_path), save_dir_test + '/' + img_name + '_lab.png')
            os.rename(str(img_path), save_dir_test + '/' + img_name + '.png')
        else:
            os.rename(str(lab_path), save_dir_trainval + '/' + img_name + '_lab.png')
            os.rename(str(img_path), save_dir_trainval + '/' + img_name + '.png')


def split_cls_data(data_root, save_root, test_rate=0.2):
    save_dir_trainval = save_root + '/trainval'
    save_dir_test = save_root + '/test'

    for cls in Path(data_root).iterdir():
        if not cls.is_dir():
            continue
        img_paths = [i for i in Path(cls).glob('*.jpeg')]
        num = len(img_paths)
        test_num = int(num * test_rate)
        random.shuffle(img_paths)
        for i, img_path in enumerate(img_paths):
            print(i, img_path)
            if i < test_num:
                save_img_dir = save_dir_test + '/' + cls.name
            else:
                save_img_dir = save_dir_trainval + '/' + cls.name
            mkdir(save_img_dir)
            shutil.copy(str(img_path), save_img_dir)


def seg_a_img(img_path):
    net = UNet(2).cuda()
    ckpt = torch.load('Results/seg-unet-h512w512-dilate0-weighting_none/unet_50.pt')
    net.load_state_dict(ckpt)
    net.eval()
    img, lab, mask, bgr = cv2_preprocess_eye(img_path, None)
    img_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
    img_tensor = img_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        out_tensor = net(img_tensor)
    out = np.array(torch.max(out_tensor.data, 1)[1].squeeze().cpu())
    # out = np.where(out > 0, 0, 255).astype('uint8')
    out = (out * mask).astype('uint8') * 255
    # print_np_info(out, 'out')
    out = np.expand_dims(out, axis=-1)
    # print(out.shape, bgr.shape)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = np.concatenate((rgb, out), axis=-1)
    # print_np_info(out, 'out')
    # print(out.shape)
    return out


def preprocess_and_segment_all_imgs(root, save_root_name):
    root_name = Path(root).name
    for img_path in Path(root).rglob('*.*'):
        img_path = str(img_path)
        save_path = img_path.replace(root_name, save_root_name)
        save_dir = Path(save_path).parent
        mkdir(save_dir)
        print(img_path)
        out = seg_a_img(img_path)
        out = Image.fromarray(out).convert('RGBA')
        out.save(save_path.replace('jpeg', 'png'))


if __name__ == '__main__':
    img, lab, mask, bgr = cv2_preprocess_eye('my_datasets/cls/ori/Mild NPDR/2965_right.jpeg', None)
    cv2.imwrite('demo_preprocess.jpg', img)

    # split_seg_dataset('my_datasets/seg/ori', 'my_datasets/seg/preprocessed')

    # aug_seg_data('my_datasets/seg/preprocessed', 'my_datasets/seg/preprocessed_aug', aug_rate=30)

    # split_aug_data('my_datasets/seg/preprocessed_aug', 'my_datasets/seg/preprocessed_aug', test_rate=0.2)

    # split_cls_data('my_datasets/cls/ori', 'my_datasets/cls/splited', test_rate=0.2)

    img = seg_a_img('my_datasets/cls/ori/Mild NPDR/2965_right.jpeg')
    img = Image.fromarray(img).convert('RGBA')
    img.save('demo_segment.png')
    rgba = Image.open('demo_segment.png')
    channels = rgba.split()
    r = np.array(channels[0])
    g = np.array(channels[1])
    b = np.array(channels[2])
    a = np.array(channels[3])
    print_np_info(r, 'r')
    print_np_info(g, 'g')
    print_np_info(b, 'b')
    print_np_info(a, 'a')
    # plt.subplots(221)
    # plt.imshow(a)
    # plt.subplots(222)
    # plt.imshow(g)
    # # plt.subplots(223)
    # plt.imshow(b)
    # plt.subplots(224)
    # plt.imshow(a)
    # plt.show()

    # preprocess_and_segment_all_imgs('my_datasets/cls/splited', 'splited_segmented')
