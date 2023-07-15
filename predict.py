import argparse
import os
import time
import math
from os import path, makedirs
from pathlib import Path

import torch.nn.functional as F
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import copy
import torch.nn as nn
from simsiam.model_factory import SimSiam

from loader import ISIC2019, CIFAR10N, CIFAR100N
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint_mine, accuracy, load_checkpoint, \
    ThreeCropsTransform
import torchvision.transforms as transforms
from queue_with_pro import *

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default='../datasets/ISIC/ISIC_2019_Training_Input', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default='./save', type=str, help='path to experiment directory')
parser.add_argument('--dataset', default='ISIC19', type=str, help='path to dataset',
                    choices=["ISIC19"])
parser.add_argument('--noise_type', default='sym', type=str, help='noise type: sym or asym', choices=["sym", "asym"])
parser.add_argument('--r', type=float, default=0.5, help='noise level')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)
parser.add_argument('--name', default='ISIC', help='save to project/name')

parser.add_argument('--arch', default='resnet18', help='Inception resnet18 model name is used for training')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--m', type=float, default=0.99, help='moving average of probbility outputs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--seed', default=123)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

parser.add_argument('--fine_tuning', type=int, default=1, help='finetuning or not')
parser.add_argument('--model_dir', type=str, default="./weight/ISICNR20%.pt", help='model weights dir')

args = parser.parse_args()
import random
import numpy

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.dataset == 'ISIC19':
    args.nb_classes = 8
    args.all = 61318




def set_model(args):
    model = SimSiam(args.m, args)
    model.cuda()
    return model


def set_loader(args):
    if args.dataset == 'ISIC19':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomResizedCrop(224, scale=(0.3, 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_set = ISIC2019(root=args.data_root, train=True,
                             transform=ThreeCropsTransform(train_transforms, train_cls_transformcon), r=args.r)

        test_data = ISIC2019(root=args.data_root, split="ISIC/test.lst", train=False, transform=test_transform)
        args.all = len(train_set.train_data)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  pin_memory=False,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8,
                                 pin_memory=False)
    elif args.dataset == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        train_set = CIFAR10N(root=args.data_root,
                             transform=ThreeCropsTransform(train_transforms, train_cls_transformcon),
                             noise_type=args.noise_type,
                             r=args.r)

        test_data = datasets.CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    return train_loader, test_loader




def validation(test_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    acc = AverageMeter('Loss', ':.4e')


    model.eval()
    end = time.time()
    with torch.no_grad():
        
        for i, (images, targets,index) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs = model.forward_test(images)
            acc2 = accuracy(outputs, targets, topk=(1,))

            # measure elapsed time
            acc.update(acc2[0].item(), images[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    

    return acc.avg

def main():
    print(vars(args))

    _, test_loader = set_loader(args)

    model = set_model(args)

    if args.fine_tuning == 1:
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

    time1 = time.time()
    val_top1_acc = validation(test_loader, model, args)
    print("Test\t time: {}\tAcc: {}".format(time.time() - time1, val_top1_acc))


if __name__ == '__main__':
    main()
