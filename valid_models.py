import os
import gc

from functools import partial
from tqdm import tqdm

import cv2
import numpy as np
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim import Adam, SGD, AdamW
from torchvision.transforms.functional import hflip, vflip, rotate

from utils.dataset_v import VesuviusDataset
from utils.image_loaders import get_train_valid_dataset_4_folds, read_images_mask_middle_layers
from utils.set_seed import set_seed
from utils.metrics import calc_cv
from torchvision.transforms.functional import hflip, vflip, rotate


from utils.triple_mit_unet_uneven import VesuviusModelTripleMIT_Uneven
from utils.gradual_warmup_scheduler_v2 import get_scheduler, scheduler_step
from utils.manet_meanpooled import MAnetMeanPooled
from utils.pooled_unet_smp import UnetMeanPooled

from torch.utils.tensorboard import SummaryWriter

class CFG:
    device = 'cuda:1'
    
    PATH_TO_DS = '../data_4_folds'
    PATH_TO_SAVE_INF = './inference'
    valid_batch_size = 1
    num_workers = 4
    
    
    criterion = smp.losses.SoftBCEWithLogitsLoss()

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

def valid_fn(valid_loader, model, criterion, img_shape, device, _rotate=False):
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

def valid_on_fold(valid_img, model, transformations, in_chans, tile_size, stride, rotate=False):
    read_images_mask = partial(read_images_mask_middle_layers, PATH_TO_DS = CFG.PATH_TO_DS, chans_idxs=in_chans, tile_size=tile_size)
    _, _, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset_4_folds(valid_img, read_images_mask, tile_size, stride, stride)
    
    valid_dataset = VesuviusDataset(valid_images, valid_masks, valid_xyxys, transformations)
    
    valid_dataloader = DataLoader(valid_dataset, 
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    valid_mask_gt = cv2.imread(os.path.join(CFG.PATH_TO_DS,  f"train/{valid_img}/inklabels.png"), 0)
    valid_mask_gt_shape = valid_mask_gt.shape
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (tile_size - valid_mask_gt.shape[0] % tile_size)
    pad1 = (tile_size - valid_mask_gt.shape[1] % tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    
    model.to(CFG.device)
    
    avg_val_loss, mask_pred = valid_fn(
        valid_dataloader, model, criterion, valid_mask_gt.shape, CFG.device, _rotate=rotate)
    
    best_dice, best_th, ths = calc_cv(valid_mask_gt, mask_pred)
    
    print(f"Avg val loss - {avg_val_loss}")
    print("THs", ths)
    print(f"best_th - {best_th}")
    print(f'best dice - {best_dice}')           
        
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_th, best_dice, mask_pred

DATASET_PATH = '/home/dmitry/Documents/KaggleCompetitions/Vesuvius/'
model_paths = [#'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_1_epoch_12_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_1_epoch_21_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_2_epoch_10_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_2_epoch_21_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_3_epoch_12_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_3_epoch_19_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_4_epoch_13_model.pth',
               #'models/mit_mean_pooling_32_ch_256/Dataset/mit_mean_pooling_32_ch_256_fold_4_epoch_13_model.pth',
               
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_1_epoch_12_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_1_epoch_23_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_2_epoch_16_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_2_epoch_25_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_3_epoch_22_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_3_epoch_31_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_4_epoch_20_model.pth',
               #'models/mit_b1_manet_32_ch/Dataset/mit_b1_manet_32_ch_fold_4_epoch_30_model.pth',
               
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_1_epoch_8_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_1_epoch_30_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_2_epoch_13_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_2_epoch_20_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_3_epoch_13_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_3_epoch_16_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_4_epoch_13_model.pth',
               'models/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven/Dataset/triple_mit_b1_b2_9_slices_from_the_middle_default_augmentations_4_folds_uneven_fold_4_epoch_20_model.pth',
               
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_1_epoch_10_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_1_epoch_21_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_2_epoch_17_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_2_epoch_22_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_3_epoch_25_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_3_epoch_30_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_4_epoch_15_model.pth',
               #'models/mit_b1_max_pooling_32/Dataset/mit_b1_max_pooling_32_fold_4_epoch_33_model.pth'
               ]
folds = [#1, 1, 2, 2, 3, 3, 4, 4,  
         #1, 1, 2, 2, 3, 3, 4, 4,  
         1, 1, 2, 2, 3, 3, 4, 4,  
         #1, 1, 2, 2, 3, 3, 4, 4
         ]

model_names = [#'mit_b2_mean_pooling_1_12',
               #'mit_b2_mean_pooling_1_21',
               #'mit_b2_mean_pooling_2_10',
               #'mit_b2_mean_pooling_2_21',
               #'mit_b2_mean_pooling_3_12',
               #'mit_b2_mean_pooling_3_19',
               #'mit_b2_mean_pooling_4_13',
               #'mit_b2_mean_pooling_4_13_',
               
               #'mit_b2_manet_mean_pooling_1_12',
               #'mit_b2_manet_mean_pooling_1_23',
               #'mit_b2_manet_mean_pooling_2_16',
               #'mit_b2_manet_mean_pooling_2_25',
               #'mit_b2_manet_mean_pooling_3_22',
               #'mit_b2_manet_mean_pooling_3_31',
               #'mit_b2_manet_mean_pooling_4_20',
               #'mit_b2_manet_mean_pooling_4_30',
               
               'triple_mit_b1_b2_1_8',
               'triple_mit_b1_b2_1_30',
               'triple_mit_b1_b2_2_13',
               'triple_mit_b1_b2_2_20',
               'triple_mit_b1_b2_3_13',
               'triple_mit_b1_b2_3_16',
               'triple_mit_b1_b2_4_13',
               'triple_mit_b1_b2_4_20',
               
               
               #'mit_b1_max_pooling_1_10',
               #'mit_b1_max_pooling_1_21',
               #'mit_b1_max_pooling_2_17',
               #'mit_b1_max_pooling_2_22',
               #'mit_b1_max_pooling_3_25',
               #'mit_b1_max_pooling_3_30',
               #'mit_b1_max_pooling_4_15',
               #'mit_b1_max_pooling_4_33'
               ]

in_chans = [#[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            #[16 + i for i in range(32)],
            
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            
            
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)],
            #[i for i in range(20, 38)]
            ]

tile_sizes = [#256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              
              448,
              448,
              448,
              448,
              448,
              448,
              448,
              448,
              
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              #256,
              ]

stride_sizes = [#128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                
                224,
                224,
                224,
                224,
                224,
                224,
                224,
                224,
                224,
                
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                #128,
                ]


def load_model(model_path, model_name, in_chans):
    best_th, best_dice = None, None
    
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    if 'mit_b1_max_pooling' in model_name:
        model = UnetMeanPooled(encoder_name = 'mit_b1', encoder_weights=None, pooling_type='max', crop_stride=2)
        
        model.load_state_dict(model_state_dict['model'])
        
        best_th = model_state_dict['best_th']
        best_dice = model_state_dict['best_dice']
        
        transformations = A.Compose([A.Normalize(
                                        mean= [0] * len(in_chans),
                                        std= [1] * len(in_chans)
                                    ),
                                    ToTensorV2(transpose_mask=True)])

    elif 'mit_b2_mean_pooling' in model_name:
        model = UnetMeanPooled(encoder_name='mit_b2', encoder_weights=None, pooling_type='mean', crop_stride=2, decoder_attention_type='scse')
        
        model.load_state_dict(model_state_dict['model'])
        
        transformations = A.Compose([A.Normalize(
                                        mean= [0] * len(in_chans),
                                        std= [1] * len(in_chans)
                                    ),
                                    ToTensorV2(transpose_mask=True)])
        
        best_th = model_state_dict['best_th']
        best_dice = model_state_dict['best_dice']
        
    elif 'triple_mit' in model_name:
        model = VesuviusModelTripleMIT_Uneven(backbone_small='mit_b1', backbone_name='mit_b2')
        
        model.load_state_dict(model_state_dict['model'])
        
        transformations = A.Compose([A.Normalize(
                                        mean= [0] * len(in_chans),
                                        std= [1] * len(in_chans)
                                    ),
                                    ToTensorV2(transpose_mask=True)])
        
        best_th = model_state_dict['best_th']
        best_dice = model_state_dict['best_dice']
        
    elif 'mit_b2_manet' in model_name:
        model = MAnetMeanPooled(encoder_name='mit_b2')
        
        model.load_state_dict(model_state_dict['model'])
        
        transformations = A.Compose([A.Normalize(
                                        mean= [0] * len(in_chans),
                                        std= [1] * len(in_chans)
                                    ),
                                    ToTensorV2(transpose_mask=True)])
        best_th = model_state_dict['best_th']
        best_dice = model_state_dict['best_dice']
    
    del model_state_dict
    gc.collect()  
    return model, transformations, best_th, best_dice
        
    


if __name__ == "__main__":
    
    logging.basicConfig(filename='./valid_models.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    logging.info("Valid Models logging")

    logger = logging.getLogger('ValidModels')
    
    for model_path, fold, model_name, in_chans, tile_size, stride_size in zip(model_paths, folds, model_names, in_chans, tile_sizes, stride_sizes):
        
        model, transformations, best_th_, best_dice_ = load_model(os.path.join(DATASET_PATH, model_path), model_name, in_chans)
        model.to(CFG.device)
        best_th, best_dice, mask = valid_on_fold(fold, model, transformations, in_chans, tile_size, stride_size, rotate=False)
        print(best_th, best_th_, best_dice, best_dice_)
        logger.info(f'{model_name} - best_th : {best_th}, best_dice : {best_dice}, best_th_old : {best_th_}, best_dice_old : {best_dice_}')
        
        np.save(os.path.join(CFG.PATH_TO_SAVE_INF, f'{model_name}_mask.npy'), mask)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()