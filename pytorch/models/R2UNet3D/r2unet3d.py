# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:46:54 2022

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
    else: raise ValueError(f'({activation}) is wrong keyword for activation')

def normalization(norm: str, n_channel):
    ''' Choose type of normalization '''
    
    if norm == 'batch': return nn.BatchNorm3d(n_channel)
    elif norm == 'instance': return nn.InstanceNorm3d(n_channel)
    else: raise ValueError(f'({n_channel}) is wrong keyword for normalization') 
    
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

class Single_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, norm:str, activation:str='relu', pad:str='same'):
        super(Single_conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=pad),
            activations(activation),
            normalization(norm, out_ch),
            )
        
    def forward(self, x):
        return self.conv(x)
        
class Recurrent_block(nn.Module):
    def __init__(self, in_ch, out_ch, norm:str, activation:str, pad:str='same', num_recurrents:int=2):
        super(Recurrent_block, self).__init__()
        
        self.num_recurrents = num_recurrents
        
        self.conv_1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=pad) 
        
        self.single_conv = Single_conv_block(out_ch, out_ch, norm, activation, pad) 
        
    def forward(self, x):
        x = self.conv_1x1(x)
        
        x1 = self.single_conv(x)
        
        for i in range(self.num_recurrents):
            x1 = x1 + x
            x1 = self.single_conv(x1)
            
        return x1
            
class R2_block(nn.Module):
    def __init__(self, in_ch, out_ch, norm:str='batch', activation:str='relu', pad:str='same', 
                 multiplier:int=2, num_stages:int=2, num_recurrents:int=2, is_residual:bool=True):
        
        super(R2_block, self).__init__()
        
        self.is_residual = is_residual
        
        assert num_stages > 1, 'num_stages should be atleast 2'
        
        # List output channels for each stage
        out_chs = [out_ch * multiplier * i for i in range(1, num_stages)] # out_channels of each stage except for 1st stage
        
        out_chs = [out_ch] + out_chs # add out_channel of the 1st stage
        
        # Create recurrent layers
        recur_in_ch = in_ch # input channel to recurrent layer
        
        recurrents = []
        
        for i in range(num_stages):
            recurrents.append(Recurrent_block(recur_in_ch, out_chs[i], norm, activation, pad, num_recurrents)
                              )
            recur_in_ch = out_chs[i] # input channel for the next recurrent layer
            
        self.recurrents = nn.Sequential(*recurrents)
        
        # Perform 1x1 conv for residual connection. 
        # self.conv_1x1_residual = nn.Conv3d(in_ch, out_ch*(2**(num_stages-1)), kernel_size=3, padding=pad)
        self.conv_1x1_residual = nn.Conv3d(in_ch, out_chs[-1], kernel_size=3, padding=pad)
        
    def forward(self, x):
        
        x1 = self.recurrents(x)

        if self.is_residual: 
            x = self.conv_1x1_residual(x)
            
            return x + x1
        
        else: return x1
            
            
# if __name__ == '__main__':
#     img = torch.rand(3, 1, 64, 64, 64)
#     r = R2_block(1, 32, num_stages=2, num_recurrents=2)
    
#     out = r(img)    
        

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

class R2UNet3D (nn.Module):
    def __init__(
            self, input_shape, base_feature, out_channel:int, multiplier: int, norm: str,  
            in_activation: str, out_activation: str, dropout:float, pad='same',
            num_stages:int=2, num_recurrents:int=2, is_residual:bool=True):
        '''
        Input
        ------
        input_shape: (tuple) shape of input. Format: Ch x H x W x D
        base_feature: (int) No. of out channels of the first convolution. (e.g. 32)
        out_channel: (int) No. of channels of final output (e.g. 2)
        multiplier: (int) Whether to double no. of out channels. It can be either 1 or 2.
        norm: (str) Normalization. Keyword: 'batch', or 'instance'
        in_activation: (str) Activation function. Keyword: 'relu', 'leaky', or 'elu'
        out_activation: (str) Activation function applied to output layer. (e.g. 'softmax' or None)
                If out_activation=None, then no activation will be applied to the output layer
        dropout: (float) Dropout
        pad: (str) Padding. Either 'same' or 'valid'
             
        Output
        -------
        Output tensor
        '''
        
        super(R2UNet3D, self).__init__()
        
        if multiplier not in [1, 2]: raise ValueError('value of multiplier can only be 1 or 2')
        
        self.dropout = nn.Dropout3d(p=dropout)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2) 
        
        # Level zero (note: suffix 'e' means encoder and 'd' means decoder)
        self.conv_e0 = R2_block(input_shape[0], base_feature, norm, in_activation, pad, multiplier, num_stages, num_recurrents, is_residual)
        
        # Level one 
        in_channel_encoder_1 = base_feature * num_stages
        self.conv_e1 = R2_block(in_channel_encoder_1, base_feature*2, norm, in_activation, pad, multiplier, num_stages, num_recurrents, is_residual)
       
        # Level two
        in_channel_encoder_2 = in_channel_encoder_1 * num_stages
        self.conv_e2 = R2_block(in_channel_encoder_2, base_feature*4, norm, in_activation, pad, multiplier, num_stages, num_recurrents, is_residual)
        
        # Level three
        in_channel_encoder_3 = in_channel_encoder_2 * num_stages
        self.conv_e3 = R2_block(in_channel_encoder_3, base_feature*8, norm, in_activation, pad, multiplier, num_stages, num_recurrents, is_residual)
        
        # Level two
        in_channel_decoder_2 = in_channel_encoder_3 * num_stages
        self.upconv_2 = up_conv(in_channel_decoder_2, base_feature*16, kernel=2, stride=2, pad=0)
        self.conv_d2 = R2_block(base_feature*(16 + 8), base_feature*8, norm, in_activation, pad, 1, num_stages, num_recurrents, is_residual) # multiplier=1 
        
        '''
        Explanation of self.base_feature*(16 + 8):
            Input to conv_d2 is concat(upconv_2, conv_e2)
            No. of output_channels in upconv_2 = self.base_feature * 16
            No. of output_channels in conv_e2 = self.base_feature * 8
            So, no. of input_channels in conv_d2 = self.base_feature (16 + 8)
        '''
        
        # Level one
        self.upconv_1 = up_conv(base_feature*8, base_feature*8, kernel=2, stride=2, pad=0)  
        self.conv_d1 = R2_block(base_feature*(8 + 4), base_feature*4, norm, in_activation, pad, 1, num_stages, num_recurrents, is_residual) # multiplier=1
        
        # Level zero
        self.upconv_0 = up_conv(base_feature*4, base_feature*4, kernel=2, stride=2, pad=0) 
        self.conv_d0 = R2_block(base_feature*(4 + 2), base_feature*2, norm, in_activation, pad, 1, num_stages, num_recurrents, is_residual) # multiplier=1
        
        # Output
        if out_activation is None:
            self.out = nn.Conv3d(base_feature*2, out_channel, kernel_size=1, stride=1, padding=pad)
        else:
            self.out = nn.Sequential(
                nn.Conv3d(base_feature*2, out_channel, kernel_size=1, stride=1, padding=pad),
                activations(out_activation)
                )
        
    def forward(self, x):
        # Level zero
        conv_e0 = self.conv_e0(x)
        
        # Level one
        maxpool_1 = self.max_pool(conv_e0)
        conv_e1 = self.conv_e1(maxpool_1)
        conv_e1 = self.dropout(conv_e1)
        
        # Level two
        maxpool_2 = self.max_pool(conv_e1)
        conv_e2 = self.conv_e2(maxpool_2)
        conv_e2 = self.dropout(conv_e2)
       
        # Level three
        maxpool_3 = self.max_pool(conv_e2)
        conv_e3 = self.conv_e3(maxpool_3)
        conv_e3 = self.dropout(conv_e3)

        # Level two
        upconv_2 = self.upconv_2(conv_e3)
        concat_2 = center_crop_and_concat(conv_e2, upconv_2)
        conv_d2 = self.conv_d2(concat_2)
        conv_d2 = self.dropout(conv_d2)
        
        # Level one
        upconv_1 = self.upconv_1(conv_d2)
        concat_1 = center_crop_and_concat(conv_e1, upconv_1)
        conv_d1 = self.conv_d1(concat_1)
        conv_d1 = self.dropout(conv_d1)
        
        # # Level zero
        upconv_0 = self.upconv_0(conv_d1)
        concat_0 = center_crop_and_concat(conv_e0, upconv_0)
        conv_d0 = self.conv_d0(concat_0)
        conv_d0 = self.dropout(conv_d0)
       
        # Output
        out = self.out(conv_d0)

        return(out)
    
# def weight_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.zeros_(m.bias)

'''
# Test model     
from torchsummary import summary
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_feature = 32
    img = torch.rand(3, 1, 64, 64, 64) # batch_size, channel, height, width
    img = img.to(device)
    input_feature = img.size()[1]
    
    input_shape = (1, 64, 64, 64)
    
    model = R2UNet3D(input_shape, base_feature, out_channel=2, multiplier=2, 
    norm='batch', in_activation='relu', out_activation=None, dropout=0.15, pad='same',
    num_stages=2, num_recurrents=2, is_residual=True)
    
    model = model.to(device)
    
    summary(model, (1,64,64,64))
    
    # model.apply(weight_init)
    
    out = model(img)
    
'''
            
    
