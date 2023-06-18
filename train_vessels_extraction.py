import os
import gc

from functools import partial
from tqdm import tqdm

import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import Dataset, dataloader

from utils.dataset_v import VesuviusDataset, VesuviusDatasetMixUP
from utils.image_loaders import get_train_valid_dataset, read_images_mask_middle_layers
from utils.image_loaders import get_train_valid_dataset_4_folds
from utils.set_seed import set_seed
from utils.metrics import calc_cv

from utils.triple_mit_unet import VesuviusModelTripleMIT_Unet
from utils.gradual_warmup_scheduler_v2 import get_scheduler, scheduler_step

from torch.utils.tensorboard import SummaryWriter

class CFG:
    
    PATH_TO_DS = "../data_drive_vessels/DRIVE/training"
    PATH_TO_SAVE = "../models/Unet_vessels_mit_b1_pretrained/"
    device = 'cuda:0'

    train_imgs = [21 + i for i in range(16)]
    test_imgs = [37 + i for i in range(4)]
    
    backbone_name='mit_b1'
    encoder_weights='imagenet' 
    activation=None
    
    epochs = 20
    
    lr = 1e-4
    size = (608, 576)
    
    criterion = smp.losses.SoftBCEWithLogitsLoss()
        
    transformations = {
        "train" : [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size[0], size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        #A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        #A.CoarseDropout(max_holes=1, max_width=int(tile_size * 0.3), max_height=int(tile_size * 0.3), 
        #                mask_fill_value=0, p=0.5),
        #A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * 3,
            std= [1] * 3
        ),
        ToTensorV2(transpose_mask=True),
    ],
    "valid" : [
        A.Resize(size[0], size[1]),
        A.Normalize(
            mean= [0] * 3,
            std= [1] * 3
        ),
        ToTensorV2(transpose_mask=True),
    ],
    "test" :[
        A.Resize(size[0], size[1]),
        A.Normalize(
            mean= [0] * 3,
            std= [1] * 3
        ),
        ToTensorV2(transpose_mask=True),
    ]}
    
    
class VesselsDataset(Dataset):
    def __init__(self, img_idxs, transformations=None):
        self.img_idxs = img_idxs
        
        self.imgs = [cv2.imread(os.path.join(CFG.PATH_TO_DS, f'images/{i}_training.tif')) for i in self.img_idxs]
        self.labels = [cv2.imread(os.path.join(CFG.PATH_TO_DS, f'labels/{i}_manual1.tiff')).max(axis=2)[:, :, None] / 255.0 for i in self.img_idxs]
        
        self.transformations = transformations
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.transformations:
            data = self.transformations(image=img, mask=label)
            
            img = data['image']
            label = data['mask']
            
        return img, label

def get_nn():
    nn = smp.Unet(
        encoder_name=CFG.backbone_name,
        encoder_weights=CFG.encoder_weights,
        in_channels = 3,
        classes=1,
        activation=CFG.activation
    )
    
    return nn

def train_nn(model, dataloader, optimizer, device=CFG.device):
    model.train()
    model.to(CFG.device)
    
    scaler = GradScaler()
    losses = []
    for img, label in tqdm(dataloader, total=len(dataloader)):
        img = img.to(device)
        label = label.to(device)        
        
        batch_size = img.shape[0]
        
        with torch.cuda.amp.autocast():
            y_pred = model(img)
            loss = CFG.criterion(y_pred, label)
        
        losses.append(loss.item() / batch_size)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        
    return np.mean(losses)


def valid_nn(model, dataloader, device):
    model.eval()
    model = model.to(device)
    
    losses = []
    
    for img, label in tqdm(dataloader, total=len(dataloader)):
        img = img.to(device)
        label = label.to(device)
        
        batch_size = img.shape[0]
        
        with torch.no_grad():
            y_pred = model(img)
            
        loss = CFG.criterion(y_pred, label)
        
        losses.append(loss.item()/ batch_size)
        
    return np.mean(losses)
        

dataset = VesselsDataset(CFG.train_imgs, A.Compose(CFG.transformations['train']))
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=3)
valid_dataset = VesselsDataset(CFG.test_imgs, A.Compose(CFG.transformations['valid']))
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True, drop_last=False)

model = get_nn()
optimizer = AdamW(model.parameters(), lr=CFG.lr)

prev_loss = np.inf

for i in range(200):
    loss = train_nn(model, dataloader, optimizer, CFG.device)
    loss_test = valid_nn(model, valid_dataloader, CFG.device)
    if loss_test < prev_loss:
        torch.save({'model' : model.state_dict(),
                    'loss' : loss_test, 
                    'loss_train' : loss}, os.path.join(CFG.PATH_TO_SAVE, 'pretrained_UNET_MIT_B1.pth'))
        prev_loss = loss_test
    print(f"Loss train : {loss}, Loss test : {loss_test}, Prev loss : {prev_loss}")


