import os
import pandas as pd
import numpy as np
import cv2
from skimage import feature
from skimage.color import rgb2gray

class Stanford_Online_Products():
    def __init__(self, root_dir):
        data_dir = os.path.join(root_dir, "Stanford_Online_Products")
        

def Stanford_Online_Products(root_dir, split='train'):
    #root_dir = os.path.dirname(__file__)
    data_dir = os.path.join(root_dir, "Stanford_Online_Products")
    if split == 'train':
        df = pd.read_csv(os.path.join(data_dir, "Ebay_train.txt"), sep=" ").sample(frac = 1)
    elif split == 'test':
        df = pd.read_csv(os.path.join(data_dir, "Ebay_test.txt"), sep=" ").sample(frac = 1)
    else:
        raise ValueError(f"Wrong split name {split}")
    return list(df["path"].map(lambda path : os.path.join(data_dir, path))), list(df["super_class_id"])

def DomainNet(root_dir, split='train'):
    data_dir = os.path.join(root_dir, "clipart")
    filelist = []
    label = []
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    filelist.append(file_path)
                    label.append(subdir)
    return filelist, label
