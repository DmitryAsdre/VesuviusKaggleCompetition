import os
import gc

from tqdm import tqdm
import numpy 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import cv2


def read_images_mask_middle_layers(fragment_id, PATH_TO_DS, chans_idxs, tile_size):
    """Takes layers only in chans_idxs"""
    images = []
    
    for i in tqdm(chans_idxs):
        image = cv2.imread(os.path.join(PATH_TO_DS, f"train/{fragment_id}/surface_volume/{i:02}.tif"), 0)

        pad0 = (tile_size - image.shape[0] % tile_size)
        pad1 = (tile_size - image.shape[1] % tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    
    images = np.stack(images, axis=2)
    
    mask = cv2.imread(os.path.join(PATH_TO_DS, f"train/{fragment_id}/inklabels.png"), 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0
    
    gc.collect()    
    return images, mask


def get_train_valid_dataset(valid_id, read_image_mask, tile_size, stride):
    """Sliding window prepare train, test sets"""
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):

        image, mask = read_image_mask(fragment_id)

        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
        
                if fragment_id == valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
    gc.collect()
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys
    
    