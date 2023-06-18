import torch
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    #SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from einops import rearrange, reduce, repeat



class SegmentationModelMeanPooled(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        n_ch = x.shape[1]
        if self.extend_chs:
            idxs =  [(i, i + 1, i+2) for i in range(0, n_ch - 3 + 1, 2)]
            idxs.extend([i, i + 5, i + 10] for i in range(0, n_ch - 11 + 1, 4))
            idxs.extend([i, i + 7, i + 14] for i in range(0, n_ch - 15 + 1, 4))
            x_prepared = [x[:, torch.LongTensor(idx), :, :] for idx in idxs]
        else:
            x_prepared = [x[:, i : i + 3, :, :] for i in range(0, n_ch - self.in_channels + 1, self.crop_stride)]
        K = len(x_prepared)
        B = x.shape[0]
        
        
        x_prepared = torch.cat(x_prepared, axis=0)

        features = self.encoder(x_prepared)
        
        
        for i in range(len(features)):
            e = features[i]
            _, c, h, w = e.shape
            e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)
            if self.pooling_type == 'mean':
                features[i] = e.mean(0)
            elif self.pooling_type == 'max':
                features[i], _ = torch.max(e, 0)
            elif self.pooling_type == 'min':
                features[i], _ = torch.min(e, 0)
                
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetMeanPooled(SegmentationModelMeanPooled):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        extend_chs = False,
        crop_stride : int = 2,
        classes: int = 1,
        pooling_type = 'mean',
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.extend_chs = extend_chs
        self.crop_stride = crop_stride
        self.in_channels = in_channels
        self.pooling_type = pooling_type
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()