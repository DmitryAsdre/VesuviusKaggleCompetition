import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VesuviusModelTripleMIT_Uneven(nn.Module):
    def __init__(self, backbone_small='mit_b0', backbone_name='mit_b2',
                       decoder_attention_type = 'scse', encoder_weights='imagenet', activation=None):
        super().__init__()
        
        self.encoder1 = smp.Unet(
            encoder_name=backbone_small,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
        self.encoder2 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
        self.encoder3 = smp.Unet(
            encoder_name = backbone_small,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=3,
            classes = 1,
            activation = activation
        )
        
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=5, padding=2)
        
        
    def forward(self, image):
        output1 = self.encoder1(image[:, :3, :, :])
        output2 = self.encoder2(image[:, 3:6, :, :])
        output3 = self.encoder3(image[:, 6:, :, :])
        
        output = torch.stack([output1, output2, output3], dim=1).squeeze(2)
        
        return self.conv(output)
    
    

class VesuviusModelTripleMIT_STACK(nn.Module):
    def __init__(self, backbone_small='mit_b0', backbone_name='mit_b2', in_chns = 15,
                       decoder_attention_type = 'scse', encoder_weights='imagenet', activation=None):
        super().__init__()
        
        self.in_conv = torch.nn.Conv2d(15, 9, kernel_size=3, padding=1)
        
        self.encoder1 = smp.Unet(
            encoder_name=backbone_small,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels = 3,
            classes=6,
            activation=activation
        )
        
        self.encoder2 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels = 3,
            classes=6,
            activation=activation
        )
        
        self.encoder3 = smp.Unet(
            encoder_name = backbone_small,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=3,
            classes = 6,
            activation = activation
        )
        
        self.conv = torch.nn.Sequential(torch.nn.BatchNorm2d(18), 
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(18, 25, kernel_size=3, padding=1), 
                                        torch.nn.BatchNorm2d(25),
                                        torch.nn.Conv2d(25, 1, kernel_size=3, padding=1))
                    #smp.Unet('resnet18', 
                    #         decoder_attention_type='scse', in_channels=18, classes=1)
        
        
    def forward(self, image):
        
        image = self.in_conv(image)
        
        output1 = self.encoder1(image[:, :3, :, :])
        output2 = self.encoder2(image[:, 3:6, :, :])
        output3 = self.encoder3(image[:, 6:, :, :])
        
        output = torch.cat([output1, output2, output3], dim=1).squeeze(2)
        
        return self.conv(output)

    