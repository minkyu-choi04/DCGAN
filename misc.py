import torch                                                        
from torch.utils.data import DataLoader                             
from torchvision import transforms                                  
import torch.optim as optim                                         
import torch.nn as nn                                               
import torch.backends.cudnn as cudnn                                
import torchvision.datasets as datasets  

import os
import numpy as np
import random

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from itertools import chain

def load_celebA(batch_size, image_size=64):
    dataset = datasets.ImageFolder(root='/export/scratch/a/choi574/DATASETS/celebA/',
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader, dataloader, 0

def load_imagenet(batch_size, image_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(root='/export/scratch/a/choi574/DATASETS/ImageNet2012/train/',
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]))
    test_data =  datasets.ImageFolder(root='/export/scratch/a/choi574/DATASETS/ImageNet2012/val',
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
            ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000


def load_lsun(batch_size, img_size=256):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='val', transform=transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.LSUN(root=os.path.expanduser('/home/libi/HDD1/minkyu/DATASETS/IMAGE/LSUN'), classes='test', transform=transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1,1.3)),
                transforms.ToTensor(),
                normalize]), target_transform=None), 
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, valid_loader, 10

def load_mnist(batch_size, img_size=32):
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                         std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='/home/libilab/a/users/choi574/DATASETS/IMAGE/mnist/', 
            train=True, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='/home/libilab/a/users/choi574/DATASETS/IMAGE/mnist/', 
            train=False, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 10

def plot_samples_from_images(images, batch_size, plot_path, filename):
    max_pix = torch.max(torch.abs(images))
    images = ((images/max_pix) + 1.0)/2.0
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.detach().cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
