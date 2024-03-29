import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch.nn as nn
import cv2
import numpy as np
import scipy


class VesuviusDataset(Dataset):
    def __init__(self, images, labels = None, valid_xyxs=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.valid_xyxs = valid_xyxs
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
        
        valid_xyxs = torch.tensor([0, 0, 0, 0])
        if self.valid_xyxs:
            valid_xyxs = torch.tensor(self.valid_xyxs[idx])         
        return image, label, valid_xyxs
    
    
class VesuviusDatasetBeta(Dataset):
    def __init__(self, images, labels = None, valid_xyxs=None, transform=None, beta_noise = False, beta_params=(3.244e+00, 5.449e+00), beta_p = 0.025):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.valid_xyxs = valid_xyxs
        self.beta_params = beta_params
        self.beta_noise = beta_noise
        self.beta_p = beta_p
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if np.random.uniform(0, 1) < self.beta_p:
            #print(image.shape, label.shape, image.dtype, label.dtype)
            label_zeros = np.where(label == 0)
            if len(label_zeros) != 0:
                beta_noize = scipy.stats.beta.rvs(a=self.beta_params[0], b=self.beta_params[1], size=image[label_zeros].shape) * 255
                beta_noize = beta_noize.astype(image.dtype)
                #print(image[label_zeros], label[label_zeros])
                image[label_zeros] = beta_noize
            #print(image.shape, label.shape, image.dtype, label.dtype)
        
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
        
        valid_xyxs = torch.tensor([0, 0, 0, 0])
        if self.valid_xyxs:
            valid_xyxs = torch.tensor(self.valid_xyxs[idx])         
        return image, label, valid_xyxs
    

class VesuviusDatasetMixUP(Dataset):
    def __init__(self, images, labels = None, valid_xyxs=None, transform=None, alpha=0.3, beta=0.3, p_mixup=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.valid_xyxs = valid_xyxs
        self.alpha = alpha
        self.beta = beta
        self.p_mixup = p_mixup
        
        
        self.mixup_idxs = []        
        for i in range(len(self.images)):
            if self.images[i].max() > 0:
                self.mixup_idxs.append(i)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
            
            if self.p_mixup and np.random.uniform() > self.p_mixup:
                gamma = np.random.beta(a=self.alpha, b=self.beta)
                _mix_idx = np.random.randint(0, len(self.mixup_idxs))
                mix_idx = self.mixup_idxs[_mix_idx]
                
                image_mix = self.images[mix_idx]
                label_mix = self.labels[mix_idx]
                
                data = self.transform(image=image_mix, mask=label_mix)
                image_mix = data['image']
                label_mix = data['mask']
            
                image = gamma*image + (1 - gamma)*image_mix
                label = gamma*label + (1 - gamma)*label_mix
            
        
        valid_xyxs = torch.tensor([0, 0, 0, 0])
        if self.valid_xyxs:
            valid_xyxs = torch.tensor(self.valid_xyxs[idx])         
        return image, label, valid_xyxs