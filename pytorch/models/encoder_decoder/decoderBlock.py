# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 18:57:04 2023

@author: mrinal
"""
import torch
from torch import nn
import torch.nn.functional as F

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    """ Here you will write your codes necessary for each decoder stage """
    
    # Here is a sample. Let's say we will do conv2d, bn, relu in each decoder stage. 
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.attention = SCSEModule(out_channels, reduction=16)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)        
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.attention(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()  
        
        encoder_channels = cfg["encoder_channels"]
        decoder_channels = cfg["decoder_channels"]
        out_channels = cfg["out_channels"]
        kernel_size = cfg["kernel_size"]
        padding = cfg["padding"]
        stride = cfg["stride"]
        dilation = cfg["dilation"]
        
        
        self.decoder_stage = len(decoder_channels)
               
        # Reverse encoder channels
        encoder_channels = encoder_channels[::-1]
        
        # Find bottleneck (head) channels
        bottleneck_channel = encoder_channels[0]
        
        # Remove bottleneck channel from the encoder channels list
        encoder_channels.pop(0) # now it does not have bottleneck channel
        
        # If length of encoder channels is less than that of decoder channels,
        # then make it same by inserting 0 in the encoder channel list. 
        
        ch_diff = abs(len(decoder_channels) - len(encoder_channels)) # difference between decoder and encoder channels length
        
        for i in range(ch_diff): encoder_channels + [0] # add None to encoder channels
        
        # Determine which stage will require concatenation (skip connection).
        # If channel value is 0, it means no concatenation, otherwise concatenation is True
        self.cat = list(map(bool, encoder_channels)) # note: here encoder list does not have bottleneck
        
        # Prepare decoder block for each stage
        decoder_blocks = []
        
        for i in range(self.decoder_stage):
            # First decoder (from bottom) takes input from bottleneck
            if i == 0:
                decoder_blocks = [
                    DecoderBlock(in_channels=bottleneck_channel+encoder_channels[i],
                                 out_channels=decoder_channels[i],
                                 kernel_size=kernel_size, padding=padding, 
                                 stride=stride, dilation=dilation)
                    ]
            
            else:
                decoder_blocks = decoder_blocks + [
                    DecoderBlock(in_channels=decoder_channels[i-1]+encoder_channels[i],
                                 out_channels=decoder_channels[i],
                                 kernel_size=kernel_size, padding=padding, 
                                 stride=stride, dilation=dilation)
                    ]
        
        
        self.blocks = nn.ModuleList(decoder_blocks)
        
        # Final output layer. Note: It does not have softmax activation. So, keep
        # it in mind while implementing the loss function
        self.output = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, *encoder_features):
       
        # Reverse features from "top to bottom" to "bottom to top"
        encoder_features = encoder_features[::-1]
        
        # Separate the bottleneck (head) from the encoder features
        x = encoder_features[0]
        
        # Remove bottleneck from the encoder feature list
        encoder_features.pop(0) # now it does not have bottleneck
        
        for i in range(self.decoder_stage):
            x = F.interpolate(x, scale_factor=2, mode="nearest") # upsample
            if self.cat[i]: x = torch.cat([x, encoder_features[i]], dim=1) # concatenate
            x = self.blocks[i](x)
            x = self.output(x)
            
        return x
            


        
        
if __name__ == '__main__':
    from torchsummary import summary
    
    cfg = {
        "encoder_channels": [32,64,128,256,512], # channels are stored from top level to bottom level
        "decoder_channels": [256,128,64,32], # channels are stored from bottom level to top level
        "out_channels": 1,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
        
        
        }

       
    net = DecoderBlock(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1)
    # x = torch.randn(20, 16, 64, 64) 
    # y = net(x)
    
    # net.to("cuda")    
    # summary(net, input_size=(16, 64, 64))    
    
    decoder = Decoder(cfg)
    decoder.to("cuda")
    # summary(decoder, input_size=(16, 64, 64))  
    
    
