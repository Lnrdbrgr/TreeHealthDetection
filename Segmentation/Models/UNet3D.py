"""
"""

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        ### Encoder ###
        self.conv1 = self.block_0(self.in_channels, 64)
        self.conv2 = self.conv_block_3D(64)
        self.conv3 = self.conv_block_3D(128)
        self.conv4 = self.conv_block_3D(256)
        self.reduce_t = self.reduce_time_dimension(512)
        self.conv5 = self.conv_block_2D(512, 1024)
        self.conv6 = self.conv_block_2D(1024, 512)

        ### Decoder ###
        self.convT1 = self.conv_block_2D_transpose(512, 512)
        self.convT2 = self.conv_block_2D_transpose(512 + 256*2, 512)
        self.convT3 = self.conv_block_2D_transpose(512 + 128*2, 256)
        self.conv7 = self.conv_block_2D(256 + 64*2, 128)
        self.conv8 = self.conv_block_2D(128, 128)
        self.conv9 = self.conv_block_2D(128, self.out_channels)

    def forward(self, x):
        """
        """
        size = x.shape[-1]
        
        ### Encoder ###
        # 3D conv
        out_1 = self.conv1(x) # out: [batch, channels: 64, t, size: 1024]
        out_2 = self.conv2(out_1) # out: [batch, channels: 128, t, size/2, size: 512]
        out_3 = self.conv3(out_2) # out: [batch, channels: 256, t, size/2, size: 256]
        out_4 = self.conv4(out_3) # out: [batch, channels: 512, t, size/2, size: 128]
        # Bring time dimension together
        out_5 = self.reduce_t(out_4) # out: [batch, channels: 512, t/2, size: 128]
        out_5 = out_5.squeeze(2) #  # out: [batch, channels: 512, size: 128]
        # 2D conv
        out_6 = self.conv5(out_5) # out: [batch, channels: 1024, size: 128]
        out_7 = self.conv6(out_6) # out: [batch, channels: 512, size: 128]

        ### Decoder ###
        out_8 = self.convT1(out_7) # out: [batch, channels: 521, size: 256]
        # skip connection
        out_3 = out_3.reshape(-1, 256*2, size//2//2, size//2//2)
        out_8 = torch.cat((out_8, out_3), dim=1)
        out_9 = self.convT2(out_8) # out: [batch, channels: 512, size: 512]
        # skip connection
        out_2 = out_2.reshape(-1, 128*2, size//2, size//2)
        out_9 = torch.cat((out_9, out_2), dim=1)
        out_10 = self.convT3(out_9) # out: [batch, channels: 256, size: 1024]
        # skip connection
        out_1 = out_1.reshape(-1, 64*2, size, size)
        out_10 = torch.cat((out_10, out_1), dim=1)
        out_11 = self.conv7(out_10) # out: [batch, channels: 128, size: 1024]
        # last conv layer
        out_12 = self.conv8(out_11)
        out_13 = self.conv9(out_12) # out: [batch, channels: num_classes, size: 1024]

        out = torch.softmax(out_13, dim=1)
        return out 

    def block_0(self, in_channels, out_channels):
        """
        in: batch, channel, t, width, height
        out: batch, out_channel, t, width, height
        """
        block = nn.Sequential()

        block.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        block.add_module('bn', nn.BatchNorm3d(out_channels))
        block.add_module('activation', nn.LeakyReLU())
        return block

    def conv_block_3D(self, in_channels):
        """
        in: batch, channel, t, width, height
        out: batch, channel*2, t, width/2, height/2
        """
        out_channels = in_channels*2
        block = nn.Sequential()

        block.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=2, padding=1))
        block.add_module('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1)))
        block.add_module('bn', nn.BatchNorm3d(out_channels))
        block.add_module('activation', nn.LeakyReLU())
        return block
    
    def reduce_time_dimension(self, in_channels):
        """
        in: batch, channel, t, width, height
        out: batch, channel, t/2, width, height
        """
        block = nn.Sequential()

        block.add_module('conv1', nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1)))
        block.add_module('bn', nn.BatchNorm3d(in_channels))
        block.add_module('activation', nn.LeakyReLU())
        return block
    
    def conv_block_2D(self, in_channels, out_channels):
        """
        in: batch, channel, width, height
        out: batch, channel*2, width, height
        """
        block = nn.Sequential()

        block.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        block.add_module('bn', nn.BatchNorm2d(out_channels))
        block.add_module('activation', nn.LeakyReLU())
        return block
    
    def conv_block_2D_transpose(self, in_channels, out_channels, upsample_factor = 2):
        """
        in: batch, channel, width, height
        out: batch, channel/2, width*2, height*2
        """
        block = nn.Sequential()
        block.add_module('con1', nn.ConvTranspose2d(in_channels, in_channels,
                                                    kernel_size=upsample_factor, stride=upsample_factor))
        block.add_module('conv2', nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        block.add_module('bn', nn.BatchNorm2d(out_channels))
        block.add_module('activation', nn.LeakyReLU())
        
        return block