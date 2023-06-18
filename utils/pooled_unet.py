import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import *

class SmpUnetDecoder(nn.Module):
	def __init__(self,
	         in_channel,
	         skip_channel,
	         out_channel,
	    ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel,]+ out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
			for i, s, o in zip(i_channel, s_channel, o_channel)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			s = skip[i]
			d = block(d, s)
			decode.append(d)

		last  = d
		return last, decode

class Config(object):
    valid_threshold = 0.80
    beta = 1
    crop_fade  = 16#32
    crop_size  = 128 #256 
    crop_depth = 5
    infer_fragment_z = [
        32-16,
        32+16,
    ]#32 slices
    dz = 0
CFG1 = Config()


class PooledResnet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.output_type = ['inference', 'loss']

        # --------------------------------
        CFG = CFG1
        self.crop_depth = CFG.crop_depth

        conv_dim = 64
        encoder_dim = [conv_dim, 64, 128, 256, 512, ]
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = resnet34d(pretrained=True, in_chans=self.crop_depth)

        self.decoder = SmpUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.logit = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)

        # --------------------------------
        self.aux = nn.ModuleList([
            nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
        ])


    def forward(self, batch):
        v = batch
        B, C, H, W = v.shape
        vv = [
            v[:, i:i + self.crop_depth] for i in range(0,C-self.crop_depth+1,2)
        ]
        K = len(vv)
        x = torch.cat(vv, 0)

        # ---------------------------------

        encoder = []
        e = self.encoder

        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x); encoder.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x); encoder.append(x)
        x = e.layer2(x); encoder.append(x)
        x = e.layer3(x); encoder.append(x)
        x = e.layer4(x); encoder.append(x)
        ##[print('encoder',i,f.shape) for i,f in enumerate(encoder)]

        for i in range(len(encoder)):
            e = encoder[i]
            _, c, h, w = e.shape
            e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)
            encoder[i] = e.mean(0)

        last, decoder = self.decoder(feature = encoder[-1], skip = encoder[:-1][::-1]  + [None])


        # ---------------------------------
        logit = self.logit(last)

        #output = {}
        #if 1:
        #    if logit.shape[2:]!=(H, W):
        #        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
        #    output['ink'] = torch.sigmoid(logit)

        return F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)

