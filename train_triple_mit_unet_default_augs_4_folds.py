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

from utils.dataset_v import VesuviusDataset
from utils.image_loaders import get_train_valid_dataset_4_folds, read_images_mask_middle_layers
from utils.set_seed import set_seed
from utils.metrics import calc_cv

from utils.triple_mit_unet import VesuviusModelTripleMIT_Unet
from utils.gradual_warmup_scheduler_v2 import get_scheduler, scheduler_step

from torch.utils.tensorboard import SummaryWriter

class CFG:
    device = 'cuda:0'
    
    PATH_TO_DS = '../data_4_folds'
    PATH_TO_SAVE = '../models/'
    exp_name = 'triple_mit_l1_9_slices_from_the_middle_default_augmentations_4_folds'
    chans_idxs = [28, 29, 30, 31, 32, 33, 34, 35, 36]
    in_chans = len(chans_idxs)
    tile_size = 448
    stride = 224
    
    train_batch_size = 12
    valid_batch_size = train_batch_size*2
    num_workers = 4
    max_grad_norm = 1_000
    use_amp = True
    
    backbone_name = 'mit_b1'
    encoder_weights = 'imagenet'
    activation = None
    
    criterion = smp.losses.SoftBCEWithLogitsLoss()
    
    max_norm = 1e3
    warmup_factor = 10
    lr = 1e-4 / warmup_factor
    epochs = 35
    
    # ============== augmentation =============
    transformations = {
        "train" : [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(tile_size, tile_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(tile_size * 0.3), max_height=int(tile_size * 0.3), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ],
    "valid" : [
        A.Resize(tile_size, tile_size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ],
    "test" :[
        A.Resize(tile_size, tile_size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg

def valid_fn(valid_loader, model, criterion, img_shape, device):
    mask_pred = np.zeros(img_shape)
    mask_count = np.zeros(img_shape)
    
    model = model.eval()
    losses = AverageMeter()
    
    for step, (images, labels, valid_xyxy) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
            
        losses.update(loss.item(), batch_size)
        
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        for i in range(batch_size):
            x1, y1, x2, y2 = valid_xyxy[i, 0].item(), valid_xyxy[i, 1].item(), valid_xyxy[i, 2].item(), valid_xyxy[i, 3].item()
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += 1

    print(f'mask count min {mask_count.min()}')
    mask_pred /= mask_count
        
    return losses.avg, mask_pred
    

def criterion(y_pred, y_true):
    return CFG.criterion(y_pred, y_true)


def train_on_fold(valid_img, writer):    
    CUR_PATH_TO_SAVE = os.path.join(CFG.PATH_TO_SAVE, CFG.exp_name, f'fold_{valid_img}')    
    os.makedirs(CUR_PATH_TO_SAVE, exist_ok=True)
    
    read_images_mask = partial(read_images_mask_middle_layers, PATH_TO_DS = CFG.PATH_TO_DS, chans_idxs=CFG.chans_idxs, tile_size=CFG.tile_size)
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset_4_folds(valid_img, read_images_mask, CFG.tile_size, CFG.stride, CFG.stride)
    
    train_dataset = VesuviusDataset(train_images, train_masks, None, A.Compose(CFG.transformations['train']))
    valid_dataset = VesuviusDataset(valid_images, valid_masks, valid_xyxys, A.Compose(CFG.transformations['valid']))

    train_dataloader = DataLoader(train_dataset, 
                              batch_size = CFG.train_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    valid_mask_gt = cv2.imread(os.path.join(CFG.PATH_TO_DS,  f"train/{valid_img}/inklabels.png"), 0)
    valid_mask_gt_shape = valid_mask_gt.shape
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    
    model = VesuviusModelTripleMIT_Unet(CFG.backbone_name, encoder_weights=CFG.encoder_weights, activation=CFG.activation)
    model = model.to(CFG.device)
    
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)
    
    for epoch in range(CFG.epochs):
        
        loss = train_fn(train_dataloader, model, criterion, optimizer, CFG.device)

        avg_val_loss, mask_pred = valid_fn(
                valid_dataloader, model, criterion, valid_mask_gt.shape, CFG.device)
        
        scheduler_step(scheduler, epoch)
        
        best_dice, best_th, ths = calc_cv(valid_mask_gt, mask_pred)
        
        print(f'Avg loss - {loss}', f"Avg val loss - {avg_val_loss}")
        print("THs", ths)
        print(f"best_th - {best_th}")
        print(f'best dice - {best_dice}')           
        
        
        if writer:
            for th in ths:
                writer.add_scalar(f'FOLD_{valid_img}/th-{th}', ths[th], epoch)
                
            writer.add_scalar(f'FOLD_{valid_img}/train_loss', loss, epoch)
            writer.add_scalar(f'FOLD_{valid_img}/valid_loss', avg_val_loss, epoch)
            writer.add_scalar(f'FOLD_{valid_img}/f_beta_best', best_dice, epoch)
            writer.add_scalar(f'FOLD_{valid_img}/best_th', best_th, epoch)
        
        if CUR_PATH_TO_SAVE:
            torch.save({'model' : model.state_dict(), 
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'best_th' : best_th,
                        'best_dice' : best_dice},
                    os.path.join(CUR_PATH_TO_SAVE, f'{CFG.exp_name}_fold_{valid_img}_epoch_{epoch}_model.pth'))
            torch.save({'preds' : mask_pred}, os.path.join(CUR_PATH_TO_SAVE, f'{CFG.exp_name}_fold_{valid_img}_epoch_{epoch}_preds.pth'))
        
        torch.cuda.empty_cache()
        gc.collect()
        

if __name__ == "__main__":
    comment = f'exp_name = {CFG.exp_name}, batch_size = {CFG.train_batch_size}, lr = {CFG.lr}'
    writer = SummaryWriter(comment=comment)
    train_on_fold(1, writer)
    train_on_fold(2, writer)
    train_on_fold(3, writer)
    train_on_fold(4, writer)
    writer.close()