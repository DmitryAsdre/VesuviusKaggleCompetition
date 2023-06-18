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
    
    
class VesuviusModelTripleMIT_Unet_15ch(nn.Module):
    def __init__(self, backbone_name='mit_b1', encoder_weights='imagenet', activation=None):
        super().__init__()
        
        self.encoder0 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
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
        
        self.encoder4 = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels = 3,
            classes=1,
            activation=activation
        )
        
        self.conv = torch.nn.Conv2d(5, 1, kernel_size=5, padding=2)
        
        
    def forward(self, image):
        output1 = self.encoder0(image[:, :3, :, :])
        output2 = self.encoder1(image[:, 3:6, :, :])
        output3 = self.encoder2(image[:, 6:9, :, :])
        output4 = self.encoder3(image[:, 9:12, :, :])
        output5 = self.encoder4(image[:, 12:15, :, :])
        
        output = torch.stack([output1, output2, output3, output4, output5], dim=1).squeeze(2)
        
        return self.conv(output)

def load_pretrained_decoder(model, path_to_decoder):
    pretrained_state_dict = torch.load(path_to_decoder)
    
    model.encoder1.decoder.load_state_dict(pretrained_state_dict)
    model.encoder2.decoder.load_state_dict(pretrained_state_dict)
    model.encoder3.decoder.load_state_dict(pretrained_state_dict)

    