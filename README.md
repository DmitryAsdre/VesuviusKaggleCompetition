# Vesuvius Kaggle Competition
This repository contains code for Vesuvius Kaggle Competition.

Kaggle Competition - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview

<b> My results</b>
- Public

![Alt text](./imgs/public_vesuvius.png)

- Private

![Alt text](./imgs/private_vesuvius.png)
## Architectures

### Triple MIT_l1 Unet
```
                -> 3 middle layers -> UNet(backbone="mit_b1") ->    
9 middle layers -> 3 middle layers -> UNet(backbone="mit_b2") ->   Conv2d -> BCELoss -> Predict
                -> 3 middle layers -> UNet(backbone="mit_b1") ->
```
### UNet Meanpooled

```
                      | mit_b2  | -> MeanPooling(Max, Min) -> |  UNet   | -> BCE(Dice loss)
                      | UNet    | -> MeanPooling(Max, Min) -> | decoder |
32 middle channels -> | encoder | -> MeanPooling(Max, Min) -> |  scse   |
                        |    |    -> MeanPooling(Max, Min) ->   |    |
                          ||      -> MeanPooling(Max, Min) ->     ||
```
### MaNet Meanpooled

```
                      | mit_b2  | -> MeanPooling(Max, Min) -> |  MaNet  | -> BCE(Dice loss)
                      | UNet    | -> MeanPooling(Max, Min) -> | decoder |
32 middle channels -> | encoder | -> MeanPooling(Max, Min) -> |         |
                        |    |    -> MeanPooling(Max, Min) ->   |    |
                          ||      -> MeanPooling(Max, Min) ->     ||
```

### UnetPlusPlus Meanpooled

```
                      |resnet101d| -> MeanPooling(Max, Min) -> |  UNet++ | -> BCE(Dice loss)
                      |  UNet    | -> MeanPooling(Max, Min) -> | decoder |
32 middle channels -> | encoder  | -> MeanPooling(Max, Min) -> |  scse   |
                        |      |    -> MeanPooling(Max, Min) ->  |    |
                           ||      -> MeanPooling(Max, Min) ->     ||
```

## Scheduler
- GradualWarmupSchedulerV2 (https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965)
## TTA
    Rotations :
        - 90
        - 180
        - 270

## Add mixup 
https://arxiv.org/abs/1710.09412

## Add 4 folds
2-nd scroll has been divided into two folds

## Final Solution
    - UnetMeanpooled 4folds
    - 32 channel from 16 to 48 with stride 2
    - TTA rotations
    - GradualWarmupSchedulerV2
    - 8 models in total (4 with best CV F_0.5 and 4 with best BCE)

