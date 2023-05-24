import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VesuviusModelTripleMIT_Unet(nn.Module):
    def __init__(self, backbone_name='mit_b1', encoder_weights='imagenet', activation=None):
        super().__init__()
        
        self.encoder1 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
        self.encoder2 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
        self.encoder3 = smp.Unet(
            encoder_name = backbone_name,
            encoder_weights=encoder_weights,
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
    
    
    