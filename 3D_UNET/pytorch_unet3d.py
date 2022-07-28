# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:35:25 2022

@author: mrinal
"""
import torch
import torch.nn as nn

def activations(activation: str):
    ''' Choose the activation function '''
    
    if activation == 'relu': return nn.ReLU(inplace=True)
    elif activation == 'leaky': return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu': return nn.ELU()
    elif activation == 'softmax': return nn.Softmax(dim=1)
    else: raise ValueError('Wrong keyword for activation')

def normalization(norm: str, n_channel):
    ''' Choose type of normalization '''
    
    if norm == 'batch': return nn.BatchNorm3d(n_channel)
    elif norm == 'instance': return nn.InstanceNorm3d(n_channel)
    else: raise ValueError('Wrong keyword for normalization') 
    
def conv_block(in_channel, stage1_out_channel, multiplier: int, norm: str, activation, pad='same'):
    ''' It performs conv-norm-activation in two stages. For the second stage,
        stage 2 output_channel = stage 1 output_channel x multiplier '''
        
    stage2_out_channel = multiplier * stage1_out_channel
    
    return nn.Sequential(
        # Stage 1
        nn.Conv3d(in_channel, stage1_out_channel, kernel_size=3, padding=pad),
        activations(activation),
        normalization(norm, stage1_out_channel),
        
        # Stage 2
        nn.Conv3d(stage1_out_channel, stage2_out_channel, kernel_size=3, padding=pad),
        activations(activation),
        normalization(norm, stage2_out_channel),        
        )

def up_conv(in_channel, out_channel, kernel, stride, pad=0):
    ''' It performs up sampling that is needed in the decoder '''
    
    return nn.ConvTranspose3d(
        in_channel, out_channel, kernel_size=kernel, stride=stride, padding=pad)    
    
def center_crop_and_concat(encoder_tensor, decoder_tensor):
    '''
    It first crops encoder tensor to match with decoder tensor size. Required when padding=0. 
    Then it concatenates the cropped encoder tensor and the decoder tensor. 
    
    Input
    ------
    encoder_tensor: Encoder tensor (source tensor)
    decoder_tensor: Decoder tensor (target tensor)
    
    Output
    ------
    Concatenation of cropped encoder tensor and decoder tensor
    '''
    
    encoder_size = encoder_tensor.size()[2:] # depth, height, and width only
    decoder_size = decoder_tensor.size()[2:] # depth, height, and width only
    
    cropped_encoder_tensor = encoder_tensor[
                :,  # batch
                :,  # channel 
                ((encoder_size[0] - decoder_size[0]) // 2):((encoder_size[0] + decoder_size[0]) // 2),
                ((encoder_size[1] - decoder_size[1]) // 2):((encoder_size[1] + decoder_size[1]) // 2),
                ((encoder_size[2] - decoder_size[2]) // 2):((encoder_size[2] + decoder_size[2]) // 2)
                ]
    
    return torch.cat([cropped_encoder_tensor, decoder_tensor], 1)

class UNet3D (nn.Module):
    def __init__(
            self, input_channel, base_feature, out_channel:int, multiplier: int, norm: str,  
            in_activation: str, out_activation: str, dropout:float, pad='same'):
        '''
        Input
        ------
        input_channel: (int) No. of input channels of the original data. (e.g. 1)
        base_feature: (int) No. of out channels of the first convolution. (e.g. 32)
        out_channel: (int) No. of channels of final output (e.g. 2)
        multiplier: (int) Whether to double no. of out channels. It can be either 1 or 2.
        norm: (str) Normalization. Keyword: 'batch', or 'instance'
        in_activation: (str) Activation function. Keyword: 'relu', 'leaky', or 'elu'
        out_activation: (str) Activation function applied to output layer. (e.g. 'softmax')
        dropout: (float) Dropout
        pad: (str) Padding. Either 'same' or 'valid'
             
        Output
        -------
        Output tensor
        '''
        
        super(UNet3D, self).__init__()
        
        self.input_channel = input_channel
        self.base_feature = base_feature
        self.out_channel = out_channel
        self.multiplier = multiplier
        self.norm = norm
        self.in_activation = in_activation
        self.out_activation = out_activation
        self.pad = pad
        
        if multiplier not in [1, 2]: raise ValueError('value of multiplier can only be 1 or 2')
        
        self.dropout = nn.Dropout3d(p=dropout)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2) 
        
        # Level zero (note: suffix 'e' means encoder and 'd' means decoder)
        self.conv_e0 = conv_block(self.input_channel, self.base_feature, self.multiplier, 
                                 self.norm, self.in_activation, self.pad)
        # Level one 
        in_channel_encoder_1 = self.base_feature * multiplier
        self.conv_e1 = conv_block(in_channel_encoder_1, self.base_feature*2, self.multiplier, 
                                 self.norm, self.in_activation, self.pad)
        # Level two
        in_channel_encoder_2 = in_channel_encoder_1 * multiplier
        self.conv_e2 = conv_block(in_channel_encoder_2, self.base_feature*4, self.multiplier, 
                                 self.norm, self.in_activation, self.pad)
        # Level three
        in_channel_encoder_3 = in_channel_encoder_2 * multiplier
        self.conv_e3 = conv_block(in_channel_encoder_3, self.base_feature*8, self.multiplier, 
                                 self.norm, self.in_activation, self.pad)
        # Level two
        in_channel_decoder_2 = in_channel_encoder_3 * multiplier
        self.upconv_2 = up_conv(in_channel_decoder_2, self.base_feature*16, kernel=2, stride=2, pad=0)
        self.conv_d2 = conv_block(self.base_feature*(16 + 8), self.base_feature*8, 1, # multiplier=1 
                                 self.norm, self.in_activation, self.pad)
        
        '''
        Explanation of self.base_feature*(16 + 8):
            Input to conv_d2 is concat(upconv_2, conv_e2)
            No. of output_channels in upconv_2 = self.base_feature * 16
            No. of output_channels in conv_e2 = self.base_feature * 8
            So, no. of input_channels in conv_d2 = self.base_feature (16 + 8)
        '''
        
        # Level one
        self.upconv_1 = up_conv(base_feature*8, self.base_feature*8, kernel=2, stride=2, pad=0)  
        self.conv_d1 = conv_block(self.base_feature*(8 + 4), self.base_feature*4, 1, # multiplier=1 
                                 self.norm, self.in_activation, self.pad)
        
        # Level zero
        self.upconv_0 = up_conv(base_feature*4, self.base_feature*4, kernel=2, stride=2, pad=0) 
        self.conv_d0 = conv_block(self.base_feature*(4 + 2), self.base_feature*2, 1, # multiplier=1 
                                 self.norm, self.in_activation, self.pad)
        
        # Output
        self.out = nn.Conv3d(self.base_feature*2, self.out_channel, kernel_size=1, stride=1, padding=self.pad)

        # self.out = nn.Sequential(
        #     nn.Conv3d(self.base_feature*2, self.out_channel, kernel_size=1, stride=1, padding=self.pad),
        #     activations(self.out_activation)
        #     )
        
    def forward(self, x):
        # Level zero
        conv_e0 = self.conv_e0(x)
        print('conv_e0: ', conv_e0.size())
        
        # Level one
        maxpool_1 = self.max_pool(conv_e0)
        conv_e1 = self.conv_e1(maxpool_1)
        conv_e1 = self.dropout(conv_e1)
        print('maxpool_1: ', maxpool_1.size(), '\nconv_e1: ', conv_e1.size())
        
        # Level two
        maxpool_2 = self.max_pool(conv_e1)
        conv_e2 = self.conv_e2(maxpool_2)
        conv_e2 = self.dropout(conv_e2)
        print('maxpool_2: ', maxpool_2.size(), '\nconv_e2: ', conv_e2.size())
        
        # Level three
        maxpool_3 = self.max_pool(conv_e2)
        conv_e3 = self.conv_e3(maxpool_3)
        conv_e3 = self.dropout(conv_e3)
        print('maxpool_3: ', maxpool_3.size(), '\nconv_e3: ', conv_e3.size())
        
        # Level two
        upconv_2 = self.upconv_2(conv_e3)
        concat_2 = center_crop_and_concat(conv_e2, upconv_2)
        conv_d2 = self.conv_d2(concat_2)
        conv_d2 = self.dropout(conv_d2)
        print('upconv_2: ', upconv_2.size(), '\nconcat_2: ', concat_2.size(), '\nconv_d2: ', conv_d2.size())

        # Level one
        upconv_1 = self.upconv_1(conv_d2)
        concat_1 = center_crop_and_concat(conv_e1, upconv_1)
        conv_d1 = self.conv_d1(concat_1)
        conv_d1 = self.dropout(conv_d1)
        print('upconv_1: ', upconv_1.size(), '\nconcat_1: ', concat_1.size(), '\nconv_d1: ', conv_d1.size())
        
        # Level zero
        upconv_0 = self.upconv_0(conv_d1)
        concat_0 = center_crop_and_concat(conv_e0, upconv_0)
        conv_d0 = self.conv_d0(concat_0)
        conv_d0 = self.dropout(conv_d0)
        print('upconv_0: ', upconv_0.size(), '\nconcat_0: ', concat_0.size(), '\nconv_d0: ', conv_d0.size())
        
        # Output
        out = self.out(conv_d0)
        print('Out: ', out.size())
        
        return(out)
    

# Test model         
if __name__ == "__main__":
    base_feature = 32
    img = torch.rand(3, 1, 64, 64, 64) # batch_size, channel, height, width
    input_shape = img.size()
    
    model = UNet3D(input_shape[1], base_feature, out_channel=2, multiplier=2, 
    norm='batch', in_activation='relu', out_activation='softmax', dropout=0.15, pad='same')
    
    out = model(img)
    

            
    
