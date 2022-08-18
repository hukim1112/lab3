from statistics import mean
from data import get_SOP
from models import EfficientB4
from utils.reproducibility import set_seed
from utils.metrics import Metric_tracker
from utils.earlystopping import EarlyStopping
from experiment.epoch import train_epoch, test_epoch

import torch
from torch.optim import AdamW
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2


#LR E-2 + EfficientB4
def exp_set1(weight_dir, log_dir, device):
    #experiment environment
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    #data
    data_loaders, class_to_name = get_SOP() #get PlantNet-300K dataset by default options.
    
    #model, optimizer, scheduler, earlystopping
    model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-2) #get your optimizer
    scheduler = None #get your scheduler. If it is None, you use a constant learning rate.
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0) #get your lr scheduler
    early_stopping = EarlyStopping(patience=7, path=weight_dir)

    #tensorboard logger and metrics.
    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics[split] = Metric_tracker(split, class_to_name, log_dir)

    # #training process
    epochs = 100
    for epoch in range(1,epochs+1):
        train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=scheduler) 
        metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
        metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

        early_stopping(metrics["val"].cal_epoch_loss(), epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
        if early_stopping.early_stop:
            print("Early stopping")
            break

    optimizer = model.load(weight_dir, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)

#LR E-3 + EfficientB4
def exp_set2(weight_dir, log_dir, device):
    #experiment environment
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    #data
    data_loaders, class_to_name = get_SOP() #get PlantNet-300K dataset by default options.
    
    
    #model, optimizer, scheduler, earlystopping
    model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = None #get your scheduler. If it is None, you use a constant learning rate.
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0) #get your lr scheduler
    early_stopping = EarlyStopping(patience=7, path=weight_dir)

    #tensorboard logger and metrics.
    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics[split] = Metric_tracker(split, class_to_name, log_dir)

    # # #training process
    # epochs = 100
    # for epoch in range(1,epochs+1):
    #     train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=scheduler) 
    #     metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
    #     metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

    #     early_stopping(metrics["val"].cal_epoch_loss(), epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break

    optimizer = model.load(weight_dir, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)

#LR E-4 + EfficientB4
def exp_set3(weight_dir, log_dir, device):
    #experiment environment
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    #data
    data_loaders, class_to_name = get_SOP() #get PlantNet-300K dataset by default options.
    
    
    #model, optimizer, scheduler, earlystopping
    model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-4) #get your optimizer
    scheduler = None #get your scheduler. If it is None, you use a constant learning rate.
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0) #get your lr scheduler
    early_stopping = EarlyStopping(patience=7, path=weight_dir)

    #tensorboard logger and metrics.
    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics[split] = Metric_tracker(split, class_to_name, log_dir)

    # #training process
    epochs = 100
    for epoch in range(1,epochs+1):
        train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=scheduler) 
        metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
        metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

        early_stopping(metrics["val"].cal_epoch_loss(), epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
        if early_stopping.early_stop:
            print("Early stopping")
            break

    optimizer = model.load(weight_dir, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)