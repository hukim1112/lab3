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
    def __init__(self, flist, sigma, transform=None, target_transform=None):
        super(EdgeDataset, self).__init__()
        self.flist = flist
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma
    def __len__(self):
        return len(self.flist)
    def __getitem__(self, index):
        try:
            image = self.load_image(index)
            edge = self.load_edge(image, self.sigma())
            category = self.load_category(index)
        except:
            print('loading error: ' + self.flist[index])
            image = self.load_image(0)
            edge = self.load_edge(image, self.sigma())
            category = self.load_category(0)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            category = self.target_transform(category)
        return image, edge, category
    def load_image(self, index):
        img = imread(self.flist[index])
        return img
    def load_edge(self, image, sigma):
        if image.shape[-1] == 3:
            img_gray = rgb2gray(image)
        else:
            img_gray = image
        return canny(img_gray, sigma=sigma)
