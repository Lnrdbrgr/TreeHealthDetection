"""
"""

import torch
import torch.nn as nn


class UNet3DPreTrained(nn.Module):
    def __init__(self, in_channels, out_channels,
                 input_size, output_size, t) -> None:
        super().__init__()

        self.encoder = EncoderBlock(in_channels=in_channels, out_channels=256)
        self.decoder = DecoderBlock(in_channels=256, out_channels=out_channels,
                                    input_size=input_size, output_size=output_size, t=t)

    def forward(self, x):
        encoder_output = self.encoder(x=x)
        decoder_output = self.decoder(x=encoder_output, orig_x=x)
        #out = torch.argmax(decoder_output, dim=1)
        out = torch.softmax(decoder_output, dim=1)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        # initialize pre-trained model
        self.pre_trained_model = torch.hub.load('facebookresearch/pytorchvideo',
                                                'slow_r50', pretrained=True)

        # initialize encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self._get_3d_conv_no_pooling(in_channels, 3))
        self.encoder.append(self._get_pretrained_model_blocks(0)) # 3 in, 64 out
        self.encoder.append(self._get_pretrained_model_blocks(1)) # 64 in, 256 out
        self.encoder.append(self._get_3d_conv_no_pooling(256, 1024))
        self.encoder.append(self._get_3d_conv_no_pooling(1024, 1024))
        self.encoder.append(self._get_3d_conv_no_pooling(1024, 512))
        self.encoder.append(self._get_3d_conv_no_pooling(512, out_channels))
    
    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        return x

    def _get_pretrained_model_blocks(self, block):
        model_block = self.pre_trained_model.blocks[block]
        # freeze weights
        for parameter in model_block.parameters():
            parameter.requires_grad = False
        return model_block

    def _get_3d_conv_no_pooling(self, in_channels, out_channels):
        # 3D conv -> batch norm -> relu
        conv_3d = nn.Sequential()
        conv_3d.append(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        conv_3d.append(nn.BatchNorm3d(out_channels))
        conv_3d.append(nn.LeakyReLU())
        return conv_3d
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 input_size, output_size, t) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        self.t = t

        # initialize decoder
        self.decoder_1 = nn.ConvTranspose3d(
            self.in_channels, out_channels=128,
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.decoder_2 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64,
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        # process skip connection
        self.decoder_skip = nn.ModuleList()
        self.decoder_skip.append(self._get_3d_conv_no_pooling(64+3, 128))
        self.decoder_skip.append(self._get_3d_conv_no_pooling(128, 64))
        self.decoder_skip.append(self._get_3d_conv_no_pooling(64, self.out_channels))
        # reduce time dimension
        self.reduce_time_dimension = nn.Conv3d(
            in_channels=self.out_channels, out_channels=self.out_channels,
            kernel_size=(self.t, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
        )
        # final layer
        self.decoder_4 = nn.Conv2d(
            in_channels=self.out_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, orig_x):
        # go through first blocks
        x = self.decoder_1(
            x,
            output_size=[1, 128, int(self.t), int(self.input_size/2),
                         int(self.input_size/2)]
        )
        x = self.decoder_2(
            x,
            output_size=[1, 64, int(self.t), int(self.input_size),
                         int(self.input_size)]
        )
        # handle skip connection
        x = torch.cat((orig_x, x), dim=1)
        for block in self.decoder_skip:
            x = block(x)
        # get rid of time dimension
        x = self.reduce_time_dimension(x)
        x = x.squeeze(2)
        # final layer
        x = self.decoder_4(x)

        return x

    def _get_3d_conv_no_pooling(self, in_channels, out_channels):
        # 3D conv -> batch norm -> relu
        conv_3d = nn.Sequential()
        conv_3d.append(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        conv_3d.append(nn.BatchNorm3d(out_channels))
        conv_3d.append(nn.LeakyReLU())
        return conv_3d
    