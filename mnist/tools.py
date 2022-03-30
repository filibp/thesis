import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from model import Generator, Discriminator
from torchvision import datasets
import matplotlib.pyplot as plt
import glob
import os
from torch import Tensor


# trainset_val_path = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/valset"
# test_normal_path  = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/healthyset"
# test_anom_path    = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/anoset"


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)

# def load_mnist(path, training_label=1, split_rate=0.8, download=True):
#     train = datasets.MNIST(path, train=True, download=download)
#     test = datasets.MNIST(path, train=False, download=download)
#     print(type(test.data))
#     _x_train = train.data[train.targets == training_label]
#     x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)),
#                                             dim=0)
#     print(len(x_train))

#     _y_train = train.targets[train.targets == training_label]
#     y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)),
#                                             dim=0)
#     print(y_train[0])
#     # show(y_train[len(y_train)-1])
    
#     x_test = torch.cat([x_test_normal,
#                         train.data[train.targets != training_label],
#                         test.data], dim=0)
#     y_test = torch.cat([y_test_normal,
#                         train.targets[train.targets != training_label],
#                         test.targets], dim=0)
#     print(x_train)
#     return (x_train, y_train), (x_test, y_test)


def load_mnist(trainset_path, image_size=275, training_label=1, split_rate=0.8):
    x_train = torch.zeros(len(os.listdir(trainset_path + '/' + 'trainset')), image_size, image_size)
    y_train = torch.zeros(len(os.listdir(trainset_path + '/' + 'trainset')))
    x_test = torch.zeros(len(os.listdir(trainset_path + '/' + 'anomalous')), image_size, image_size)
    y_test = torch.zeros(len(os.listdir(trainset_path + '/' + 'anomalous')))
    for i, folder in enumerate(os.listdir(trainset_path)):
        for j, image in enumerate(os.listdir(trainset_path + '/' + folder)):
            if j % 1000 == 0:
                print(j/1000)
            # x_train = torch.cat((x_train, torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)), 0)
            image = torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)
            if folder == 'trainset':
                x_train[j] = image
                # print(len(torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)))
                y_train[j] = i 
                # print(len(y_train))
            else:
                x_test[j] = image
                y_test[j] = i

    return  (x_train, y_train), (x_test, y_test)
