from operator import mod
import os
import time
from torchvision import datasets, transforms, models
import os
import argparse
import torch
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', type=bool, default=0)
    parser.add_argument('--train_aug', type=bool, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_learning_rate', type=float, default=0.0001)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='results/cls')
    parser.add_argument('--data_root', type=str, default='data/eyepace_kaggle/split_preprocess')
    parser.add_argument('--resume_path', type=str, default='')
    args = parser.parse_args()
    return args

def mkdir(d):
    if not os.path.exists(d): os.makedirs(d)

def vgg_finetune(num_classes, pretrained=True, requires_grad=False):
    ##加载基于VGG19的模型
    model = models.vgg19(pretrained=pretrained) 

    for param in model.parameters():
        param.requires_grad = requires_grad

    #修改最后一层
    number_features = model.classifier[6].in_features 
    features = list(model.classifier.children())[:-1] # 移除最后一层
    features.extend([torch.nn.Linear(number_features, num_classes)]) 
    model.classifier = torch.nn.Sequential(*features)
    print(model)
    return model

def cls_train_val(args):
    mkdir(args.save_dir)

    writer = SummaryWriter(args.save_dir)

    transform_aug = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(10, expand=False),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if args.train_aug:
        train_transform = transform_aug
    else:
        train_transform = transform

    train_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'test'), transform=transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    len_train_dataset = len(train_dataset)
    len_val_dataset = len(val_dataset)
    print('len_train_dataset:{}, len_val_dataset:{}'.format(len_train_dataset, len_val_dataset))

    len_train_dataloader = len(train_dataloader)
    len_val_dataloader = len(val_dataloader)
    print('len_train_dataloader:{}, len_val_dataloader:{}'.format(len_train_dataloader, len_val_dataloader))

    train_class_names = train_dataset.classes
    val_class_names = val_dataset.classes
    print('train_class_names', train_class_names)
    print('val_class_names', val_class_names)
    assert train_class_names == val_class_names

    num_classes = len(train_class_names)

    model = vgg_finetune(num_classes, pretrained=True, requires_grad=True)

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
        time0 = time.time()
        model.train()
        losses = []
        for epoch_iter, (img, lab) in enumerate(train_dataloader):
            img = img.to(device)
            lab = lab.to(device)
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss = loss.item()
            if epoch_iter % 2 == 0:
                print('Epoch:[{}/{}] Iter:[{}/{}] Loss:{}'.format(epoch, args.total_epoch, epoch_iter, len_train_dataloader, iter_loss))
            losses.append(iter_loss)
        lr_scheduler.step()
        train_loss = sum(losses) / len(losses)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        model.eval()
        losses = []
        for img, lab in val_dataloader:
            with torch.no_grad():
                img = img.to(device)
                lab = lab.to(device)
                out = model(img)
                loss = criterion(out, lab)
            losses.append(loss.item())
        val_loss = sum(losses) / len(losses)
        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        time1 = time.time()
        epoch_time = time1 - time0

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                "model": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, args.save_dir + '/cls.pt')
            print('Got min_val_loss:{} epoch_time:{}'.format(min_val_loss, epoch_time))

def cls_predict_imgs(model, img_transform, img_dir, save_dir):
    mkdir(save_dir)

    paths = [i for i in Path(img_dir).rglob('*.jpeg')]
    for path in paths:
        print(str(path))
        image = Image.open(str(path)).convert('RGB')
        img_tensor = img_transform(image).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            out = model(img_tensor)
            print(out)

def cls_cal_metrics(model, img_transform, img_dir, save_dir):
    mkdir(save_dir)

    GT, PD = [], []
    image_dataset = datasets.ImageFolder(img_dir, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)
    for img, lab in dataloader:
        with torch.no_grad():
            img = img.to(device)
            lab = lab.to(device)
            out = model(img)
            out = torch.max(out, 1)[1]
            # print(lab, out)
            GT.append(lab[0])
            PD.append(out[0])
    GT = np.array(GT).astype('uint8')
    PD = np.array(PD).astype('uint8')
    print(GT.shape, PD.shape)
    acc = accuracy_score(GT, PD)
    p = precision_score(GT, PD, average='weighted')
    r = recall_score(GT, PD, average='weighted')
    f1 = f1_score(GT, PD, average='weighted')
    conf = confusion_matrix(GT, PD)
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}
    print(metrics)

def cls_test(args):
    checkpoint = torch.load(args.save_dir + '/cls.pt')
    model = vgg_finetune(2, pretrained=False, requires_grad=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img_dir = args.data_root + '/test'
    save_dir = args.save_dir

    cls_cal_metrics(model, transform, img_dir, save_dir)

    # cls_predict_imgs(model, img_transform, img_dir, save_dir)

if __name__ == '__main__':
    args = get_args()
    
    if not args.test_only:
        cls_train_val(args)
    
    cls_test(args)

