# Vesuvius Kaggle Competition
This repository contains code for Vesuvius Kaggle Competition.

Kaggle Competition - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview

## Triple MIT_l1 Unet
```
                -> 3 middle layers -> UNet(backbone="mit_l1") ->    
9 middle layers -> 3 middle layers -> UNet(backbone="mit_l1") ->   Conv2d -> BCELoss -> Predict
                -> 3 middle layers -> UNet(backbone="mit_l1") ->
```

## Add mixup 
https://arxiv.org/abs/1710.09412

## Add 4 folds
2-nd scroll has been divided into two folds

## Train stride : 200