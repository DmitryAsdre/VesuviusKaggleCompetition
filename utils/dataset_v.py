import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch.nn as nn
import cv2


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