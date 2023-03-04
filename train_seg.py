import os
import json
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from dataset import SegDataset, get_class_weights
from choices import choose_net, get_criterion, get_optimizer, get_lr_scheduler
from predictor import eval_dataset_full, predict_images
from pathlib import Path


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--net-name", type=str)
    parser.add_argument("--save-suffix", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--dilate", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--train-set", type=str)
    parser.add_argument("--val-set", type=str)
    parser.add_argument("--test-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--opt-name", type=str, default='adam')
    parser.add_argument("--sch-name", type=str, default='warmup_poly')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--pt-stride", type=int, default=1)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--eval", type=bool, default=True)
    return parser.parse_args()


def train(args):
    train_set = SegDataset(args.train_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                           dilate=args.dilate)
    print('Length of train_set:', len(train_set))
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_class_weights = get_class_weights(train_dataloader, out_channels=args.out_channels, weighting=args.weighting)
    if args.eval:
        val_set = SegDataset(args.val_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                             dilate=0)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_class_weights = get_class_weights(val_dataloader, out_channels=args.out_channels, weighting=args.weighting)
    else:
        val_dataloader, val_class_weights = None, None

    test_set = SegDataset(args.test_set, args.out_channels, appoint_size=(args.height, args.width), dilate=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_class_weights = get_class_weights(test_loader, args.out_channels, args.weighting)

    # Prepare save dir
    save_dir = './Results/' + args.save_suffix + '-' + args.net_name + '-h' + str(train_set[0][0].shape[1]) + 'w' \
               + str(train_set[0][0].shape[2]) + '-dilate' + str(args.dilate) + '-weighting_' + str(args.weighting)
    print('Save dir is:{}  Input size is:{}'.format(save_dir, train_set[0][0].shape))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Prepare network
    writer = SummaryWriter(save_dir)
    val_dicts = []
    net = choose_net(args.net_name, args.out_channels).cuda()
    train_criterion = get_criterion(args.out_channels, class_weights=train_class_weights)
    optimizer = get_optimizer(net, args.opt_name)

    steps = len(train_dataloader)
    lr_scheduler = get_lr_scheduler(optimizer, max_iters=args.epoch * steps, sch_name=args.sch_name)

    # Begin to train
    iter_cnt = 0
    for epo in range(args.epoch):
        net.train()
        for batch_id, (batch_data, batch_label, _) in enumerate(train_dataloader):
            if args.out_channels == 1:
                batch_label = batch_label.float()  # 逻辑损失需要label的类型和data相同，均为float，而不是long
            else:
                batch_label = batch_label.squeeze(1)  # 交叉熵label的类型采用默认的long，但需要去除C通道维
            # print(batch_data.shape, batch_label.shape)
            iter_cnt += 1
            output = net(batch_data.cuda())
            loss = train_criterion(output, batch_label.cuda())
            iter_loss = loss.item()
            print('Epoch:{} Batch:[{}/{}] Train loss:{}'.format(epo + 1, str(batch_id + 1).zfill(3), steps,
                                                                round(iter_loss, 4)))
            writer.add_scalar('Train loss', iter_loss, iter_cnt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None and args.opt_name != 'adam':
                lr_scheduler.step()

        if args.eval:
            v_loss, (miou, pa) = eval_dataset_full(net.eval(), args.out_channels, val_dataloader,
                                                   class_weights=val_class_weights, save_dir=None)
            writer.add_scalar('Val loss', v_loss, epo + 1)
            writer.add_scalar('Val miou', miou, epo + 1)
            writer.add_scalar('Val pa', pa, epo + 1)
            val_dict_tmp = {}
            val_dict_tmp.setdefault('epoch', epo + 1)
            val_dict_tmp.setdefault('loss', v_loss)
            val_dict_tmp.setdefault('miou', miou)
            val_dict_tmp.setdefault('pa', pa)
            val_dicts.append(val_dict_tmp)

        if (epo + 1) == args.epoch or (epo + 1) % args.pt_stride == 0 or epo == 0:
            save_file = save_dir + '/' + args.net_name + '_{}.pt'.format(epo + 1)
            torch.save(net.state_dict(), save_file)
            print('Saved checkpoint:', save_file)

    writer.close()
    with open(save_dir + '/val_log.json', 'w') as f2:
        json.dump(val_dicts, f2, indent=2)

    net.eval()
    with torch.no_grad():
        eval_dataset_full(net, args.out_channels, test_loader, class_weights=test_class_weights, save_dir=save_dir)
        # args.test_images = [str(i) for i in Path(args.test_set).rglob('*.jpg')]
        args.pt_dir = ''
        predict_images(net, args, dst_size=(512, 512), save_dir=save_dir)


def do_train(args):
    with torch.cuda.device(args.gpu):
        train(args)


def get_choices(args, task):
    sizes = [(512, 512)]
    weightings = ['none']

    if task == 0:
        dilates = [0]
        args.train_set = 'data/KolektorSDD_split/train'
        args.val_set = 'data/KolektorSDD_split/test'
        args.test_set = 'data/KolektorSDD_split/test'
        args.test_images = 'data/KolektorSDD_split/test'

    return args, sizes, dilates, weightings


def search_train(args):
    args.out_channels = 2
    args.epoch = 50
    args.batch_size = 2
    args.gpu = 0

    train_net_names = ['unet'] # 'enet', 'unet', 'segnet', 'lanenet0508'
    save_suffix = 'seg'
    task = 0
    args, sizes, dilates, weightings = get_choices(args, task=task)

    for net_name in train_net_names:
        for size in sizes:
            for dilate in dilates:
                for weighting in weightings:
                    args.weighting = weighting
                    args.dilate = dilate
                    args.net_name = net_name
                    args.height = size[0]
                    args.width = size[1]
                    args.save_suffix = save_suffix

                    do_train(args)


if __name__ == "__main__":
    args = get_train_args()
    search_experiment = True

    if search_experiment:
        search_train(args)
    else:
        do_train(args)
