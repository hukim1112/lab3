from config.path import PATH
from torch.utils.data import DataLoader
from .load_flist import StanfordOnlineProducts, OfficeHome
from .EdgeData import EdgeDataset, EdgeInputDataset
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

default_transforms = {
    'train': A.Compose([
        A.Resize(height=380, width=380),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'val': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'test': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
}

def get_SOP(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    SOP = StanfordOnlineProducts(root_dir=PATH["SOP"])
    splits = SOP.get_flist()

    data_loaders = {}
    for split in splits:
        X,Y = splits[split]
        dataset = EdgeDataset(flist=X, labels=Y, transform = transforms[split], target_transform=lambda x : SOP.label2class()[x] ,sigma=2.0)
        if split == "train":
            class_to_name = SOP.class2label()
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name

def get_SOPE(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    SOP = StanfordOnlineProducts(root_dir=PATH["SOP"])
    splits = SOP.get_flist()

    data_loaders = {}
    for split in splits:
        X,Y = splits[split]
        dataset = EdgeInputDataset(flist=X, labels=Y, transform = transforms[split], target_transform=lambda x : SOP.label2class()[x] ,sigma=2.0)
        if split == "train":
            class_to_name = SOP.class2label()
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name