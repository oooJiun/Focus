import glob
import random
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
class ImageDataset(Dataset):
    def __init__(self, root, img_transform_=None, mask_transform_=None):
        self.img_transform = transforms.Compose(img_transform_)
        self.mask_transform = transforms.Compose(mask_transform_)
        self.files = sorted(glob.glob(root + '/*.*'))
        
    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(8)
        # if(len(self.files[index].split('/')[-1].split('.')[0])>4): 
        #     name = int(self.files[index].split('/')[-1].split('.')[0][4:8])
        # else:
        #     name = int(self.files[index].split('/')[-1].split('.')[0])
        img = Image.open(self.files[index]).convert('RGB')

        if(len(self.files[index].split('/')[-1].split('.')[0])>4):
            # mask_img = Image.open(os.path.join("/home/oscar/Desktop/0301_dpdd_seg/dpdd_seg_mask_copy", self.files[index].split('/')[-1])).convert('L')
            # mask_img = Image.open(os.path.join("./source_map", self.files[index].split('/')[-1][:9]+"bmp")).convert('L')
            mask_img = Image.open(os.path.join("./source_map", self.files[index].split('/')[-1])).convert('L')

            # print("!", self.files[index])
            # print("?", os.path.join("/home/oscar/Desktop/0301_dpdd_seg/dpdd_seg_mask_copy", self.files[index].split('/')[-1]))
            gt_img = Image.open(os.path.join("/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source", self.files[index].split('/')[-1])).convert('RGB')
            return self.img_transform(img), self.mask_transform(mask_img), self.img_transform(gt_img)
        else:
            # print("!", self.files[index])
            return self.img_transform(img)

    def __len__(self):
        return len(self.files)#,len(self.files1)



def Get_dataloader(path,batch):
    #Image.BICUBIC
    img_transform_ = [ transforms.Resize((256,256)),
                transforms.ToTensor()
    ]
    
    mask_transform_ = [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]

    train_dataloader = DataLoader(
        ImageDataset(path, img_transform_=img_transform_, mask_transform_ = mask_transform_),
        batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
    return train_dataloader




class ImageDataset_test(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        name = int(self.files[index].split('/')[-1].split('.')[0][4:8])
        img = Image.open(self.files[index]).convert('RGB')

        return self.transform(img),name

    def __len__(self):
        return len(self.files)

def Get_dataloader_test(path,batch):
    transforms_ = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(
        ImageDataset_test(path, transforms_=transforms_),
        batch_size=batch, shuffle=False, num_workers=2, drop_last=True)

    return train_dataloader


