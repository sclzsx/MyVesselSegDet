import os
import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange
from model import ViT
from utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', type=bool, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_learning_rate', type=float, default=0.001)
    parser.add_argument('--total_epoch', type=int, default=3000)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='results/cls')
    parser.add_argument('--data_root', type=str, default='/home/SENSETIME/sunxin/3_datasets/eyepace_kaggle/split')
    parser.add_argument('--resume_path', type=str, default='')
    args = parser.parse_args()
    return args

def mkdir(d):
    if not os.path.exists(d): os.makedirs(d)

def cls_train_val(args):
    mkdir(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    data_transforms = {
        'train': transforms.Compose([
            #    transforms.CenterCrop(input_shape),
                transforms.Resize((512, 512)),
            transforms.ToTensor(),
            #    transforms.Normalize([0.5, 0.5, 0.5], std)
        ]),
        'test': transforms.Compose([
            #    transforms.CenterCrop(input_shape),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            #    transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(args.data_root, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=args.batch_size,
            shuffle=True, num_workers=4
        )
        for x in ['train', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print(dataset_sizes)

    class_names = image_datasets['train'].classes
    print(class_names)

    ##加载基于VGG19的模型
    model = torchvision.models.vgg19(pretrained=True) 

    for param in model.parameters():
        param.requires_grad = True

    #修改最后一层
    number_features = model.classifier[6].in_features 
    features = list(model.classifier.children())[:-1] # 移除最后一层
    features.extend([torch.nn.Linear(number_features, len(class_names))]) 
    model.classifier = torch.nn.Sequential(*features) 

    model = model.to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_learning_rate)
    milestones = [i * args.total_epoch // 5 for i in range(1, 5)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    if os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']

    min_val_loss = 10000
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.total_epoch):
        model.train()
        losses = []
        for img, lab in dataloaders['train']:
            img = img.to(device)
            lab = lab.to(device)
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)
        writer.add_scalar('train_loss', avg_loss, global_step=epoch)
        print(f'[{epoch}/{args.total_epoch}], average train loss is {avg_loss}.')

        model.eval()
        losses = []
        for img, lab in dataloaders['test']:
            with torch.no_grad():
                img = img.to(device)
                lab = lab.to(device)
                out = model(img)
                loss = criterion(out, lab)
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('val_loss', avg_loss, global_step=epoch)
        if avg_loss < min_val_loss:
            min_val_loss = avg_loss
            checkpoint = {
                "model": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, args.save_dir + '/cls.pt')
            print(f'Got min_val_loss: {min_val_loss}. Updated checkpoint.')


if __name__ == '__main__':
    args = get_args()
    
    cls_train_val(args)
