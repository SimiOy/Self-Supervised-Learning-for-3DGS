# Author: Xu Yan
# Year: 2019
# Code adapted from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master


import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from data_loader import ModelNetDataLoader
from models import inplace_relu, FeatureFusionNetwork, CustomLRScheduler
from models.network_utils import evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    # Architecture
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    # Model params
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--beta', type=float, default=3.0, help='weight for the cross-modality loss')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--num_iterations', default=60000, type=int, help='number of iterations in training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    # Optimizers
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD')
    parser.add_argument('--decay_steps', type=int, default=250, help='decay steps for learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate for learning rate')

    # Data Loader Args
    parser.add_argument('--point_dir', type=str, required=True, help='3D Gaussian Splats Dir')
    parser.add_argument('--img_dir', type=str, required=True, help='Renders/Images Dir')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', help='use normals')
    parser.add_argument('--use_scale_and_rotation', action='store_true', help='use scale and rotation')
    parser.add_argument('--use_colors', action='store_true', help='use colors')
    parser.add_argument('--furthest_point_sample', action='store_true', help='use furthest point sampling')

    return parser.parse_args()


def main(args):
    def log_string(str, out=False):
        logger.info(str)
        if out:
            print(str, flush=True)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE LOG DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # point_dir = 'C:/Gaussian-Splatting/gaussian-splatting/output/modelNet10/'
    # img_dir = 'C:/ResearchProject/datasets/modelnet10/ModelNet10_captures'

    train_dataset = ModelNetDataLoader(point_dir=args.point_dir, img_dir=args.img_dir, args=args, split='train')
    test_dataset = ModelNetDataLoader(point_dir=args.point_dir, img_dir=args.img_dir, args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)

    '''MODEL LOADING'''
    pointnet_model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('train_model.py', str(exp_dir))

    no_features = 3
    if args.use_normals:
        no_features += 3
    if args.use_colors:
        no_features += 44
    if args.use_scale_and_rotation:
        no_features += 7

    pointnet = pointnet_model.get_model(no_features=no_features, normal_channel=args.use_normals)
    # pointnet_criterion = pointnet_model.get_loss()
    pointnet.apply(inplace_relu)

    # use pretrained model?
    # resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet18 = models.resnet18()
    resnet18.fc = nn.Identity()  # Remove the last fully connected layer

    fusion_network = FeatureFusionNetwork()

    if not args.use_cpu:
        pointnet = pointnet.cuda()
        resnet18 = resnet18.cuda()
        fusion_network = fusion_network.cuda()
        # pointnet_criterion = pointnet_criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        pointnet.load_state_dict(checkpoint['pointnet_state_dict'])
        resnet18.load_state_dict(checkpoint['resnet18_state_dict'])
        fusion_network.load_state_dict(checkpoint['fusion_network_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    params = list(pointnet.parameters()) + list(resnet18.parameters()) + list(fusion_network.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    else:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        lr_scheduler = CustomLRScheduler(optimizer, args.decay_steps, args.decay_rate)

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0
    triplet_loss = nn.TripletMarginLoss(margin=2.0)
    cross_entropy_loss = nn.CrossEntropyLoss()

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string(f'Epoch {epoch + 1} ({epoch + 1}/{args.epoch}):', out=True)
        pointnet.train()
        resnet18.train()
        fusion_network.train()

        for batch_id, (points, img1, img2, img3, y1, y2, y3) in tqdm(enumerate(trainDataLoader, 0),
                                                                     total=len(trainDataLoader), smoothing=0.9):
            if global_step >= args.num_iterations:
                break

            optimizer.zero_grad()
            points = points.transpose(2, 1)
            if not args.use_cpu:
                points = points.cuda()
                img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
                y1, y2, y3 = y1.cuda(), y2.cuda(), y3.cuda()

            # used to automatically cast between float16 and float32
            with torch.cuda.amp.autocast():
                fp, _ = pointnet(points)
                fi1 = resnet18(img1)
                fi2 = resnet18(img2)
                fi3 = resnet18(img3)

                hat_y1 = fusion_network(fi1, fp)
                hat_y2 = fusion_network(fi2, fp)
                hat_y3 = fusion_network(fi3, fp)

                L_triplet = triplet_loss(fi1, fi2, fi3)

                L_cross1 = cross_entropy_loss(hat_y1, y1)
                L_cross2 = cross_entropy_loss(hat_y2, y2)
                L_cross3 = cross_entropy_loss(hat_y3, y3)
                L_cross = L_cross1 + L_cross2 + L_cross3

                L_self = L_triplet + args.beta * L_cross

            # used to automatically cast between float16 and float32
            scaler.scale(L_self).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1

            log_string(f'Epoch {epoch + 1} Total Loss: {L_self.item()}, '
                       f'CV Loss: {L_triplet.item()}, CM Loss: {L_cross.item()}')

        # Evaluate model on test set
        log_string(f'Started evaluation:', out=True)
        cm_accuracy, mpd_positive, mpd_negative = evaluate_model(pointnet, resnet18, fusion_network, testDataLoader, num_repeat=1)

        log_string(f'Epoch {epoch + 1} CM Accuracy: {cm_accuracy}', out=True)
        log_string(
            f'Epoch {epoch + 1} CV Mean Pair Distance: Positive Pairs: {mpd_positive}, Negative Pairs: {mpd_negative}',
            out=True)

        # Save the model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = str(checkpoints_dir) + f'/checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'pointnet_state_dict': pointnet.state_dict(),
                'resnet18_state_dict': resnet18.state_dict(),
                'fusion_network_state_dict': fusion_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            log_string(f'Saved checkpoint: {checkpoint_path}', out=True)

        if global_step >= args.num_iterations:
            break

    logger.info('End of training...')
    torch.save({
        'epoch': epoch + 1,
        'pointnet_state_dict': pointnet.state_dict(),
        'resnet18_state_dict': resnet18.state_dict(),
        'fusion_network_state_dict': fusion_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(checkpoints_dir) + '/best_model.pth')

    # Evaluate model on test set
    log_string(f'Started evaluation:', out=True)
    cm_accuracy, mpd_positive, mpd_negative = evaluate_model(pointnet, resnet18, fusion_network, testDataLoader, num_repeat=5)

    log_string(f'Epoch {epoch + 1} CM Accuracy: {cm_accuracy}', out=True)
    log_string(
        f'Epoch {epoch + 1} CV Mean Pair Distance: Positive Pairs: {mpd_positive}, Negative Pairs: {mpd_negative}',
        out=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
