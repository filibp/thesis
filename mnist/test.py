import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from fanogan.train_wgangp import train_wgangp

from model import Generator, Discriminator
from tools import SimpleDataset
from torchvision import datasets
import matplotlib.pyplot as plt
import glob
import os
from torch import Tensor


def show(img):
    # img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for "
                            "discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400,
                    help="interval betwen image samples")
parser.add_argument("--training_label", type=int, default=0,
                    help="label for normal images")
parser.add_argument("--split_rate", type=float, default=0.8,
                    help="rate of split for normal training data")
parser.add_argument("--seed", type=int, default=None,
                    help="value of a random seed")
opt = parser.parse_args()


def load_dataset(path, training_label=1, split_rate=0.8, download=True):
    train = datasets.MNIST(path, train=True, download=download)
    test = datasets.MNIST(path, train=False, download=download)
    print(type(test.data))
    _x_train = train.data[train.targets == training_label]
    x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)),
                                            dim=0)
    print(len(x_train))

    _y_train = train.targets[train.targets == training_label]
    y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)),
                                            dim=0)
    print(y_train[0])
    # show(y_train[len(y_train)-1])
    
    x_test = torch.cat([x_test_normal,
                        train.data[train.targets != training_label],
                        test.data], dim=0)
    y_test = torch.cat([y_test_normal,
                        train.targets[train.targets != training_label],
                        test.targets], dim=0)
    print(x_train)
    return (x_train, y_train), (x_test, y_test)


trainset_path     = "C:/Users/Filip/DTU/msc/codes/f-AnoGAN/imagess"
trainset_val_path = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/valset"
test_normal_path  = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/healthyset"
test_anom_path    = "C:/Users/Filip/DTU/msc/codes/officialfanogan/f-AnoGAN/images/anoset"


def get_files(data_set):
        if data_set == 'train_normal':
            return glob(os.path.join(trainset_path, "*.png"))
        if data_set == 'valid_normal':
            return glob(os.path.join(trainset_val_path, "*.png"))
        elif data_set == 'test_normal':
            return glob(os.path.join(test_normal_path, "*.png"))
        elif data_set == 'test_anom':
            return glob(os.path.join(test_anom_path, "*.png"))


def load_dataset2():
    x_train = Tensor([])
    y_train = Tensor([])
    x_test = Tensor([])
    y_test = Tensor([])
    for folder in os.listdir(trainset_path):
        for image in os.listdir(trainset_path + '/' + folder):
            # x_train = torch.cat((x_train, torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)), 0)
            if folder == 'trainset':
                print(folder)
                # x_train = torch.cat((x_train, torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)), 0)
                # y_train = torch.cat((y_train, Tensor([i]))) 
            # else:
                # print(folder)
                # x_test = torch.cat((x_test, torchvision.io.read_image(trainset_path + '/' + folder + '/' + image)), 0)
                # y_test = torch.cat((y_test, Tensor([i]))) 

    return  (x_train, y_train), (x_test, y_test)


def load_mnist(path, training_label=1, split_rate=0.8, download=True):

    x_train = [None] * len(os.listdir(trainset_path + '/' + 'trainset'))
    y_train = [None] * len(os.listdir(trainset_path + '/' + 'trainset'))
    x_test = [None] * len(os.listdir(trainset_path + '/' + 'anomalous'))
    y_test = [None] * len(os.listdir(trainset_path + '/' + 'anomalous'))
    for i, folder in enumerate(os.listdir(trainset_path)):
        print(folder)
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

    return  (Tensor(x_train), Tensor(y_train)), (Tensor(x_test), Tensor(y_test))

# import os
# import pandas as pd
# from torchvision.io import read_image

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# my_dataset = CustomDataSet(img_folder_path, transform=trsfm)
# train_loader = data.DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
#                                num_workers=4, drop_last=True)


# (x_train, y_train), _ = load_dataset("datasette",
#                                     training_label=opt.training_label,
#                                     split_rate=opt.split_rate)

# load_mnist()
import datetime

datastamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs("results" + datastamp + "/images", exist_ok=True)
# print(x_train)