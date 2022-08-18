from genericpath import isdir
from logging import root
import os, random, pickle
from os.path import dirname, join, exists, isdir
import pandas as pd
import numpy as np
import cv2
from skimage import feature
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

class StanfordOnlineProducts():
    def __init__(self, root_dir, label_filter=None):
        self.data_dir = os.path.join(root_dir, "Stanford_Online_Products")
        self.label_filter = label_filter
        self.class2label = self._class2label()
        self.label2class = self._label2class()
    def get_flist(self):
        X, Y = [], []
        for sub_name in os.listdir(self.data_dir):
            if isdir(join(self.data_dir, sub_name)):
                sub_dir = join(self.data_dir, sub_name)
                label = self.subdir2label(sub_dir)
                fps = [join(sub_dir, f) for f in os.listdir(sub_dir)]
            X+=fps
            Y+=[label]*len(fps)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=614, stratify=Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=614, stratify=Y_train)
        splits = {'train' : [X_train, Y_train], 'test' : [X_test, Y_test], 'val' : [X_val, Y_val]}
        if self.label_filter is not None:
            for split in splits:
                X,Y = splits[split]
                X,Y = self.label_filtering(X,Y)
                splits[split] = [X,Y]
        return splits
    def subdir2label(self, sub_dir):
        return sub_dir.split('/')[-1][:-6]
    def path2label(self, path):
        return dirname(path).split('/')[-1][:-6]
    def label_filtering(self, X,Y):
        filtered = list(map(lambda y : y in self.label_filter, Y))
        X = [x for x,f in zip(X,filtered) if f==True]
        Y = [y for y,f in zip(Y,filtered) if f==True]
        return X,Y
    def _class2label(self):
        class2labels = []
        for sub in os.listdir(self.data_dir):
            if os.path.isdir(join(self.data_dir, sub)):
                class2labels.append(sub.split('/')[-1][:-6])
        if self.label_filter is not None:
            class2labels = self.label_filter
        class2labels.sort()
        return class2labels
    def _label2class(self):
        class2labels = self._class2label()
        label2class = {}
        for i, label in enumerate(class2labels):
            label2class[label] = i
        return label2class

'''
SOP = StanfordOnlineProducts(root_dir="/home/files/datasets/StanfordOnlineProducts")
splits = SOP.get_flist()
X_train, Y_train = splits['test']
label2class = SOP.label2class
# class2label :['bicycle', cabinet', 'chair','coffee_maker','fan','kettle','lamp','mug','sofa','stapler','table','toaster']
dataset = EdgeDataset(flist=X_train, labels=Y_train, sigma=2.0)
image, edge, label = dataset.__getitem__(0)
'''

class OfficeHome():
    def __init__(self, root_dir, type='Product', label_filter=None):
        self.root_dir = root_dir
        self.label_filter = label_filter
        self.type = type
    def get_flist(self):
        X, Y = [], []
        for label in os.listdir(join(self.root_dir, self.type)):
            dir_path = join(self.root_dir, self.type, label)
            fps = [join(dir_path, f) for f in os.listdir(dir_path)]
            X+=fps
            Y+=[label]*len(fps)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=614, stratify=Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=614, stratify=Y_train)
        splits = {'train' : [X_train, Y_train], 'test' : [X_test, Y_test], 'val' : [X_val, Y_val]}
        if self.label_filter is not None:
            for split in splits:
                X,Y = splits[split]
                X,Y = self.label_filtering(X,Y)
                splits[split] = [X,Y]
        return splits

    def label_filtering(self, X,Y):
        filtered = list(map(lambda y : y in self.label_filter, Y))
        X = [x for x,f in zip(X,filtered) if f==True]
        Y = [y for y,f in zip(Y,filtered) if f==True]
        return X,Y
    def class2label(self):
        class2labels = []
        for sub in os.listdir(self.data_dir):
            if os.path.isdir(join(self.data_dir, sub)):
                class2labels.append(sub.split('/')[-1][:-6])
        if self.label_filter is not None:
            class2labels = self.label_filter
        class2labels.sort()
        return class2labels
    def label2class(self):
        class2labels = self.class2label()
        label2class = {}
        for i, label in enumerate(class2labels):
            label2class[label] = i
        return label2class
'''
OH = OfficeHome(root_dir="/home/files/datasets/OfficeHome/OfficeHomeDataset_10072016")
splits = OH.get_flist()
X_train, Y_train = splits['train']
label2class = OH.label2class
dataset = EdgeDataset(flist=X_train, labels=Y_train, sigma=2.0)
image, edge, label = dataset.__getitem__(0)

'''

class DomainNet():
    def __init__(self, root_dir):
        self.data_dir = os.path.join(root_dir, "clipart")
    def get_flist(self, split='train'):
        filelist = []
        label = []
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        filelist.append(file_path)
                        label.append(subdir)
        return filelist, label
