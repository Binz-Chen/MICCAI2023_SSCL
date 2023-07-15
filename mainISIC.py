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

from loader import ISIC2019
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint_mine, accuracy, load_checkpoint, \
    ThreeCropsTransform
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from queue_with_pro import *

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default='../datasets/ISIC/ISIC_2019_Training_Input', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default='./save', type=str, help='path to experiment directory')
parser.add_argument('--dataset', default='ISIC19', type=str, help='path to dataset',
                    choices=["ISIC19", "ILSVRC-2012"])
parser.add_argument('--noise_type', default='sym', type=str, help='noise type: sym or asym', choices=["sym", "asym"])
parser.add_argument('--r', type=float, default=0.5, help='noise level')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)
parser.add_argument('--name', default='ISIC', help='save to project/name')

parser.add_argument('--arch', default='Inception', help='Inception resnet18 model name is used for training')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
parser.add_argument('--start_epochs', type=int, default=30, help='number of using pseudo-label epochs')
parser.add_argument('--warm_epochs', type=int, default=0, help='number of using 对比')

parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--m', type=float, default=0.99, help='moving average of probbility outputs')
parser.add_argument('--tau', type=float, default=0.4, help='contrastive threshold (tau)')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--all', default=50000, type=int, help='t')
parser.add_argument('--t', default=0.5, type=float, help='t')

parser.add_argument('--lama', default=8.0, type=float, help='lambda for class term')
parser.add_argument('--lamb', default=2.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--lamc', default=2.0, type=float, help='lambda for Information Bottleneck')
parser.add_argument('--type', default='gce', type=str, help='ce or gce loss', choices=["ce", "gce"])
parser.add_argument('--beta', default=0.6, type=float, help='gce parameter')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

parser.add_argument('--mix_up', default=0, type=int, help='use mixup or not.')
parser.add_argument('--loss_mix', default=1, type=int, help='use mixup loss or not.')
parser.add_argument('--fine_tuning', type=int, default=0, help='finetuning or not')
parser.add_argument('--model_dir', type=str, default="", help='model weights dir')
parser.add_argument('--mode', default=2, type=int, help='0为不使用任何策略   2为使用伪标签策略')
parser.add_argument('--GCE_mode', default=0, type=int, help='0为使用旧的   1为使用类平衡')

args = parser.parse_args()
import random
import numpy

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.dataset == 'ISIC19':
    args.nb_classes = 8
    args.all = 61318


# class GCE_loss(nn.Module):
#     def __init__(self, args):
#         super(GCE_loss, self).__init__()
#         self.q = args.beta
#         self.weight = numpy.array([2, 1, 1, 2, 2, 5, 5, 2])
#         if args.GCE_mode == 1:
#             print(self.weight)

#     def forward(self, outputs, targets):
#         # targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
#         if args.GCE_mode == 0:
#             pred = F.softmax(outputs, dim=1)
#             pred_y = torch.sum(targets * pred, dim=1)
#             pred_y = torch.clamp(pred_y, 1e-4)
#             final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
#         else:
#             # weight = numpy.array([2/3, 1/5, 0.8, 4, 1, 20, 20, 5])
#             weight = torch.from_numpy(self.weight).cuda()
#             weight = torch.matmul(targets.double(), weight.double().t())
#             pred = F.softmax(outputs, dim=1)
#             pred_y = torch.sum(targets * pred, dim=1)
#             pred_y = torch.clamp(pred_y, 1e-4)
#             final_loss = torch.mean(weight*(1.0 - pred_y ** self.q) / self.q, dim=0)
#         return final_loss

class GCE_loss(nn.Module):
    def __init__(self, args):
        super(GCE_loss, self).__init__()
        self.q = args.beta

    def forward(self, outputs, targets):
        # targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = F.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss

if args.type == 'ce':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = GCE_loss(args)


from torchsummary import summary
def set_model(args):
    model = SimSiam(args.m, args)
    model.cuda()
    
    summary(model,[(3,224,224),(3,224,224),(3,224,224)])
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

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=8,
                                 pin_memory=False)

    return train_loader, test_loader


## Input interpolation functions
def mix_data(x, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(5, 1)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, index, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    # if loss_a.item() < 0:
    #     # print(pred)
    #     # print(y_a)
    return lam * loss_a + (1 - lam) * loss_b


def train(train_loader, model, criterion, optimizer, epoch, args, change_index, queue):
    global lam
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    change_nums = 0
    end = time.time()

    outputs_item = torch.zeros(args.all,args.nb_classes).cuda()
    # outputs_item = torch.zeros(args.all).cuda()
    for j, (images, targets, index) in enumerate(train_loader):
        change_index_batch = change_index[index]
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        # lam = np.random.beta(5, 1)
        # images_list, t0_a, t0_b = mixup_aligned(images[0], targets, lam)
        # _, _, outputs_ce = model(images_list, images[1], images[2])

        ##Interpolated inputs Mixup操作
        if args.mix_up == 1:
            # images[2], mix_index, lam = mix_data(images[2])
            images[0], _, _ = mix_data(images[0])
            images[1], _, _ = mix_data(images[1])
            # with torch.no_grad():  # no gradient to keys
            #     lam = np.random.beta(5, 1)
            #     images[0], t0_a, t0_b = mixup_aligned(images[0], targets, lam)
        # compute output
        p1, z2, outputs = model(images[0], images[1], images[2])
        # p2, z1, _ = model(images[1], images[0], images[2])

        # avoid collapsing and gradient explosion
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)
        # p2 = torch.clamp(p2, 1e-4, 1.0 - 1e-4)
        # z1 = torch.clamp(z1, 1e-4, 1.0 - 1e-4)

        # queue.enqueue_dequeue(p1, p2, z1, z2, outputs)
        queue.enqueue_dequeue(p1, p1, z2, z2, outputs)
        queue_p1, queue_p2, queue_z1, queue_z2, queue_outputs = queue.get()

        contrast_1 = torch.matmul(p1, queue_z2.t())  # B X B
        # contrast_2 = torch.matmul(p2, queue_z1.t())  # B X B
        bsz1 = p1.size(0)
        bsz2 = queue_p1.size(0)
        contrast_1 = -contrast_1 * torch.zeros(bsz1, bsz2).fill_diagonal_(1).cuda() + (
            (1 - contrast_1).log()) * torch.ones(bsz1, bsz2).fill_diagonal_(0).cuda()
        contrast_logits = 2 + contrast_1

        # contrast_2 = -contrast_2 * torch.zeros(bsz1, bsz2).fill_diagonal_(1).cuda() + (
        #     (1 - contrast_2).log()) * torch.ones(bsz1, bsz2).fill_diagonal_(0).cuda()
        # contrast_logits_2 = 2 + contrast_2

        targets = targets.long().cuda()
        if args.dataset != 'ChestXray14':
            targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
            soft_targets = torch.softmax(outputs, dim=1)
            queue_soft_targets = torch.softmax(queue_outputs, dim=1)
        else:
            soft_targets = torch.softmax(outputs)
            queue_soft_targets = torch.sigmoid(queue_outputs)
        contrast_mask = torch.matmul(soft_targets, queue_soft_targets.t()).clone().detach()
        contrast_mask.fill_diagonal_(1)
        # 选择大于tau的对
        pos_mask = (contrast_mask >= args.tau).float()

        # 计算contrast_mask
        contrast_mask = contrast_mask * pos_mask
        contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)
        loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)

        # 计算contrast_mask_2
        # loss_ctr_2 = (contrast_logits_2 * contrast_mask).sum(dim=1).mean(0)


        # 对称预测
        # loss_ctr = (loss_ctr + loss_ctr_2) / 2

        start_epoch = args.start_epochs
        if epoch >= start_epoch and args.mode != 0:
            c = soft_targets * targets.long().cuda()
            outputs_item[index] = c
            # outputs_item[index] = c.sum(dim=1) / targets.sum(dim=1)
            if epoch >= start_epoch + 1:
                ### 构造伪标签标签
                p_lable = torch.zeros(targets.size(0), args.nb_classes).cuda()
                _, p_index = soft_targets.topk(8, dim=1, largest=False)
                # print(soft_targets)
                for i in range(targets.size(0)):
                    p_lable[i][p_index[i]] = soft_targets[i][p_index[i]]
                targets = change_index_batch * targets - (1 - change_index_batch) * p_lable
                # print(targets)
        # mix_targets = targets[mix_index]
        if args.loss_mix == 1:
            # loss_ce = mixup_criterion(criterion, outputs, targets.float().cuda(), mix_targets.float().cuda(), lam)
            loss_ce = criterion(outputs, targets.long().cuda())
        elif args.loss_mix == 2:
            loss_ce = criterion(outputs, targets.long().cuda())

        loss_IB = F.kl_div(F.log_softmax(z2, dim=-1), F.softmax(p1, dim=-1), reduction='sum') #+  F.kl_div(F.log_softmax(z1, dim=-1), F.softmax(p2, dim=-1), reduction='sum')
        if epoch < args.warm_epochs:
            loss = args.lamb *loss_ce + args.lamc * loss_IB
        else:
            loss = args.lama * loss_ctr + args.lamb *loss_ce + args.lamc * loss_IB

        if j == 1:
            print(" loss_ce=" + str(loss_ce.item()) + " loss_ctr=" + str(
                loss_ctr.item()) + " loss_IB=" + str(loss_IB.item()) + " loss=" + str(loss.item()))

        # compute gradient and do SGD step
        # print("compute gradient and do SGD step...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # print("measure elapsed time...")
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    print("change_nums=" + str(args.all - change_index.sum()))
    return losses.avg, outputs_item, queue


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_embedding(data, label, title,epoch):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图，经过验证111正合适，尽量不要修改
	# 遍历所有样本
	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 10),s=1)
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.title(title, fontsize=14)
        
	f = plt.gcf()
	f.savefig('./ISIC19/SSCL0.5/{}.png'.format(epoch))
	f.clear()
    

    
	# 返回值
	return fig

def validation(test_loader, model, epoch, args,best_acc):
    batch_time = AverageMeter('Time', ':6.3f')
    acc = AverageMeter('Loss', ':.4e')

    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=500)

    features = torch.rand(0).cuda()
    labels = torch.rand(0).cuda()

    model.eval()
    end = time.time()
    with torch.no_grad():
        
        for i, (images, targets,index) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                # targets = targets.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs = model(images)
            acc2 = accuracy(outputs, targets, topk=(1,))

            feature = model.forward_TSNE(images)


            features = torch.cat([features,feature])
            labels = torch.cat([labels,targets])


            # measure elapsed time
            acc.update(acc2[0].item(), images[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    # if epoch % 100 == 0 or epoch == args.epochs-1 or acc.avg > best_acc:
    #     reslut = tsne.fit_transform(features.cpu().numpy())
    #     fig = plot_embedding(reslut, labels.cpu().numpy(), 't-SNE Embedding of digits',epoch)

    return acc.avg


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """
    创建运行保存目录
    """
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def get_change_index(args, outputs_item):
    change_index = torch.zeros(args.all, 1).cuda()
    num = numpy.array([3675,10581,2660,653,2108,126,140,457])
    # print(outputs_item)
    # print(outputs_item.size())
    for i in range(args.nb_classes):
        _, indices = outputs_item[:,i].topk(int((1-args.t) * num[i]),dim=0)
        change_index[indices] = 1
    print(change_index.sum())
    return change_index

# def get_change_index(args, outputs_item):
#     change_index = torch.ones(args.all, 1).cuda()
#     _, indices = outputs_item.topk(int(args.t * args.all), largest=False)
#     change_index[indices] = 0
#     return change_index

def main():
    print(vars(args))

    train_loader, test_loader = set_loader(args)

    model = set_model(args)

    if args.fine_tuning == 1:
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

    start_epoch = 0

    save_dir = str(increment_path(
        Path('runs/train/' + args.name + '/')))
    w = save_dir + '/weights'
    Path(w).mkdir(parents=True, exist_ok=True)  # make dir
    # routine
    best_acc = 0.0

    # outputs_item = torch.zeros(args.all).cuda()
    outputs_item = torch.zeros(args.all,args.nb_classes).cuda()
    queue = queue_with_pro(args)

    with open(save_dir + '/log.txt', 'a') as f:
        f.write(args.__str__() + "\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_optim = epoch

        adjust_learning_rate(optimizer, epoch_optim, args)
        print("Training...")

        # train for one epoch
        time0 = time.time()
        change_index = get_change_index(args, outputs_item)
        train_loss, outputs_item, queue = train(train_loader, model, criterion, optimizer, epoch, args, change_index,
                                                queue)
        print("Train \tEpoch:{}/{}\ttime: {}\tLoss: {}".format(epoch, args.epochs, time.time() - time0, train_loss))

        time1 = time.time()
        val_top1_acc = validation(test_loader, model, epoch, args)
        print("Test\tEpoch:{}/{}\t time: {}\tAcc: {}".format(epoch, args.epochs, time.time() - time1, val_top1_acc))
        if val_top1_acc > best_acc:
            weight_name = w + '/best.pt'
            save_checkpoint_mine(epoch, model, optimizer, val_top1_acc, weight_name, "保存模型")
        best_acc = max(best_acc, val_top1_acc)
        # if (epoch % 50) == 0:
        weight_name = w + '/last.pt'
        save_checkpoint_mine(epoch, model, optimizer, val_top1_acc, weight_name, "保存模型")
        with open(save_dir + '/log.txt', 'a') as f:
            f.write(
                'epoch: {}\t train_loss: {}\t val_top1_acc: {} time: {}\n'.format(
                    epoch, train_loss, val_top1_acc, time.time() - time0))
        # scheduler.step()
    print(
        'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {'
        '} \t best_acc: {}\tlast_acc: {}\n'.format(
            args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed, best_acc,
            val_top1_acc))
    with open(save_dir + '/log.txt', 'a') as f:
        if args.type == 'ce':
            f.write(
                'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: ce \t seed: {} \t '
                'best_acc: {}\tlast_acc: {}\n'.format(
                    args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.seed, best_acc, val_top1_acc))
        elif args.type == 'gce':
            f.write(
                'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {'
                '} \t best_acc: {}\tlast_acc: {}\n'.format(
                    args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed, best_acc,
                    val_top1_acc))


if __name__ == '__main__':
    main()
