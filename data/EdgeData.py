import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
#from scipy.misc import imread
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, flist, labels, sigma, transform=None, target_transform=None):
        super(EdgeDataset, self).__init__()
        self.flist = flist
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma
    def __len__(self):
        return len(self.flist)
    def __getitem__(self, index):
        try:
            image = self.load_image(index)
            edge = self.load_edge(image, self.sigma)
            label = self.load_label(index)
        except:
            print('loading error: ' + self.flist[index])
            image = self.load_image(0)
            print(image.shape, self.sigma)
            edge = self.load_edge(image, self.sigma)
            label = self.load_label(0)
        if self.target_transform is not None:
            label = self.target_transform(label)        
        if self.transform: #transform must be albumentation's tranform.
            transformed = self.transform(image=image, mask=edge.astype(np.uint8))
            return {"image" : transformed['image'], "edge" : transformed['mask'], "label" : label}
        return {"image" : image, "edge" : edge, "label" : label}
        
    def load_image(self, index):
        img = imread(self.flist[index], pilmode='RGB')
        return img
    def load_label(self, index):
        return self.labels[index]
    def load_edge(self, image, sigma):
        if image.shape[-1] == 3:
            img_gray = rgb2gray(image)
        else:
            img_gray = image
        return canny(img_gray, sigma=sigma)

class EdgeInputDataset(torch.utils.data.Dataset):
    def __init__(self, flist, labels, sigma, transform=None, target_transform=None):
        super(EdgeInputDataset, self).__init__()
        self.flist = flist
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma
    def __len__(self):
        return len(self.flist)
    def __getitem__(self, index):
        try:
            image = self.load_image(index)
            edge = self.load_edge(image, self.sigma)
            label = self.load_label(index)
        except:
            print('loading error: ' + self.flist[index])
            image = self.load_image(0)
            print(image.shape, self.sigma)
            edge = self.load_edge(image, self.sigma)
            label = self.load_label(0)
        edge = gray2rgb(edge).astype(np.uint8)*255
        if self.transform: #transform must be albumentation's tranform.
            transformed = self.transform(image=edge)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return {"image" : transformed['image'], "label" : label}
    def load_image(self, index):
        img = imread(self.flist[index], pilmode='RGB')
        return img
    def load_label(self, index):
        return self.labels[index]
    def load_edge(self, image, sigma):
        if image.shape[-1] == 3:
            img_gray = rgb2gray(image)
        else:
            img_gray = image
        return canny(img_gray, sigma=sigma)