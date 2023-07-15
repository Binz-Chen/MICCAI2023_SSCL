import numpy as np
import json
import os
import torch
import sys
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
import math
import random
import csv


def corrupted_labels(targets, r=0.4, noise_type='sym'):
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                  8: 8}  # class transition for asymmetric noise
    size = int(len(targets) * r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0, 9))
            elif noise_type == 'asym':
                noisy_label.append(transition[targets[i]])
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x

class CIFAR10N(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(self, root, transform, noise_type, r):
        super(CIFAR10N, self).__init__(root, download=True)
        self.noise_targets = corrupted_labels(self.targets, r, noise_type)
        self.transform = transform

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.noise_targets[index], self.targets[index]
        img = Image.fromarray(img)

        im_1 = self.transform(img)

        return im_1, target, true_target, index
    
    def change_data(self, index,p_targets):
        index = index.int().cpu()
        index=index.tolist()
        index = torch.tensor(index).numpy()
        # self.noise_targets= self.targets
        # self.targets = self.noise_targets
        self.noise_targets = p_targets
        # self.data, self.noise_targets, self.targets =np.array(self.data)[index], np.array(self.targets)[index], np.array(self.targets)[index]
        self.data, self.noise_targets, self.targets =np.array(self.data)[index], np.array(self.noise_targets)[index], np.array(self.targets)[index]
        print(len(self.data))


class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # ---- Open file, get image paths and labels
        fileDescriptor = open(pathDatasetFile, "r")
        # ---- get into the loop
        line = True
        while line:

            line = fileDescriptor.readline()

            # --- if not empty
            if line:
                lineItems = line.split()

                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        # print("index = "+str(index) + " listImagePaths.len="+str(len(self.listImagePaths))+"self.listImageLabels="+str(len(self.listImageLabels)))
        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform is not None: imageData = self.transform(imageData)

        a = 0

        return imageData, imageLabel, a, index

    def __len__(self):

        return len(self.listImagePaths)


def corrupted_labels100(targets, r=0.4, noise_type='sym'):
    size = int(len(targets) * r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0, 99))
            elif noise_type == 'asym':
                noisy_label.append((targets[i] + 1) % 100)
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x


class CIFAR100N(CIFAR100):
    """CIFAR100 Dataset.
    """

    def __init__(self, root, transform, noise_type, r):
        super(CIFAR100N, self).__init__(root, download=True)
        self.noise_targets = corrupted_labels100(self.targets, r, noise_type)
        self.transform = transform

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.noise_targets[index], self.targets[index]
        img = self.data[index]
        img = Image.fromarray(img)

        im_1 = self.transform(img)

        return im_1, target, true_target, index
    
    def change_data(self, index,p_targets):
        index = index.int().cpu()
        index=index.tolist()
        index = torch.tensor(index).numpy()
        # self.noise_targets= self.targets
        # self.targets = self.noise_targets
        self.noise_targets = p_targets
        # self.data, self.noise_targets, self.targets =np.array(self.data)[index], np.array(self.targets)[index], np.array(self.targets)[index]
        self.data, self.noise_targets, self.targets =np.array(self.data)[index], np.array(self.noise_targets)[index], np.array(self.targets)[index]
        print(len(self.data))




def corrupted_labels_x(targets, r=0.4, x=8):
    size = int(len(targets) * r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            nl = random.randint(0, x - 1)
            while nl ==  targets[i]:
                nl = random.randint(0, x - 1)
            noisy_label.append(random.randint(0, x - 1))
        else:
            noisy_label.append(targets[i])
    noisy_label = np.array(noisy_label)
    return noisy_label


import numpy


class ISIC2019(Dataset):

    def __init__(self, root='', train=True, split='ISIC/train.lst', r=0.1,
                 groundtruth_file='ISIC/ISIC_2019_Training_GroundTruth.csv',
                 transform=None, temporal_label_file='label_isic.txt'):
        self.root = root    
        self.transform = transform
        self.train = train
        self.data_list = open(split).read().splitlines()
        self.num = len(self.data_list)
        self.path = temporal_label_file
        num = numpy.array([0, 0, 0, 0, 0, 0, 0, 0])
        num_old = numpy.array([0, 0, 0, 0, 0, 0, 0, 0]) 
        # now load the picked numpy arrays
        if train:
            self.train_data = []
            self.train_labels = []
            with open(groundtruth_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[0] in self.data_list:
                        label = row.index("1.0") - 1
                        num_old[label] = num_old[label] + 1
                        weight = numpy.array([1, 1, 1, 3, 2, 5, 5, 4])
                        for i in range(weight[label]) :
                            num[label] = num[label] + 1
                            self.train_data += [os.path.join(self.root, row[0] + '.jpg')]
                            self.train_labels += [label]
                        num[label] = num[label]+1
                        self.train_data += [os.path.join(self.root, row[0] + '.jpg')]
                        self.train_labels += [label]
            self.noise_targets = corrupted_labels_x(self.train_labels, r, x=8)
            # print("train images nums:" + str(len(self.noise_targets)))
            self.num = len(self.noise_targets)
            # print(num)
            # print(num_old)
        else:
            self.test_data = []
            self.test_labels = []
            with open(groundtruth_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[0] in self.data_list:
                        label = row.index("1.0") - 1
                        num[label] = num[label] + 1
                        self.test_data += [os.path.join(self.root, row[0] + '.jpg')]
                        self.test_labels += [label]
            # print("test images nums:" + str(len(self.test_data)))
            # print(num)

    def __getitem__(self, index):
        if self.train:
            image_path, target = self.train_data[index], self.noise_targets[index]
        else:
            image_path, target = self.test_data[index], self.test_labels[index]

        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return self.num
