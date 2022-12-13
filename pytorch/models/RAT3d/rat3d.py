# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:51:29 2022

RATUNet3D - Tailed residual attention UNet3D

@author: mrinal
"""
import torch
import torch.nn as nn

def activations(activation: str):
    ''' Choose the activation function '''
    
    if activation == 'relu': return nn.ReLU(inplace=True)
    elif activation == 'leaky': return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu': return nn.ELU()
    elif activation == 'sigmoid': return nn.Sigmoid()
    elif activation == 'softmax': return nn.Softmax(dim=1)
    else: raise ValueError('Wrong keyword for activation')

def normalization(norm: str, n_channel):
    ''' Choose type of normalization '''
    
    if norm == 'batch': return nn.BatchNorm3d(n_channel)
    elif norm == 'instance': return nn.InstanceNorm3d(n_channel)
    elif norm == None: pass # do nothing
    else: raise ValueError('Wrong keyword for normalization') 
    
class conv_block(nn.Module):
    def __init__(self, in_channel, stage1_out_channel, multiplier: int, norm: str, activation, pad='same'):
        super(conv_block, self).__init__()
        
        ''' It performs conv-norm-activation in two stages. For the second stage,
            stage 2 output_channel = stage 1 output_channel x multiplier. Finally,
            it applies residual connection. '''
        
        # Get output channels for the 2nd stage
        stage2_out_channel = multiplier * stage1_out_channel
        
        # Apply conv --> normalization--> activation (cna)
        self.double_cna = nn.Sequential(
            # Stage 1
            nn.Conv3d(in_channel, stage1_out_channel, kernel_size=3, padding=pad),
            normalization(norm, stage1_out_channel),
            activations(activation),
            
            # Stage 2
            nn.Conv3d(stage1_out_channel, stage2_out_channel, kernel_size=3, padding=pad),
            normalization(norm, stage2_out_channel),    
            # No activation right now                
            )
        
        # Resnet Connection
        self.shortcut = nn.Sequential(
            nn.Conv3d(stage2_out_channel, stage2_out_channel, kernel_size=1, stride=1, padding='same', bias=True),
            normalization(norm, stage2_out_channel),
            )
        
        # Here add two tensors
        
        self.res_activation =  activations(activation)
        
    def forward(self, x):
        x1 = self.double_cna(x)
        x2 = self.shortcut(x1)
        x3 = x2 + x1
        x3 = self.res_activation(x3)
        
        return x3

def gating_signal(in_channel, out_channel, activation: str, norm=None):

    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding='same'),
        normalization(norm, out_channel),
        activations(activation),
        )

class attention_block(nn.Module):
    def __init__(self, F_int, shape):
        super(attention_block, self).__init__()
        
        self.F_int = F_int
        
        self.theta_x = nn.Conv3d(F_int, F_int, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.phi_g = nn.Conv3d(F_int, F_int, kernel_size=1, padding='same')
        
        self.upsample_g = nn.ConvTranspose3d(F_int, F_int, kernel_size=3, stride=1, padding=1)
        
        # Here is the concatenation step
        
        self.act_xg = nn.ReLU(inplace=True)
        
        self.psi = nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.sigmoid_xg = nn.Sigmoid()
        
        self.upsample_psi = nn.Upsample(scale_factor=2)
        
        # Here is repeat elements. upsample_psi has 1 ch only. Repeat it F_int times.
        
        # Here is multiplication between x and upsample_psi
        
        self.result = nn.Conv3d(F_int, shape, kernel_size=1, padding='same')
        
        self.result_bn = nn.BatchNorm3d(shape)
                
    def forward(self, g, x):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        upsample_g = self.upsample_g(phi_g)        
        concat_xg = upsample_g + theta_x
        act_xg = self.act_xg(concat_xg)
        psi = self.psi(act_xg)
        sigmoid_xg = self.sigmoid_xg(psi)
        upsample_psi = self.upsample_psi(sigmoid_xg)
        upsample_psi = torch.repeat_interleave(upsample_psi, self.F_int, dim=1)
        y = upsample_psi * x
        result = self.result(y)
        result_bn = self.result_bn(result)
        
        return result_bn
           
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


class RATUNet3D(nn.Module):
    def __init__(
            self, input_shape, base_feature, out_channel:int, multiplier: int, norm: str,  
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
        out_activation: (str) Activation function applied to output layer. (e.g. 'softmax' or None)
                If out_activation=None, then no activation will be applied to the output layer
        dropout: (float) Dropout
        pad: (str) Padding. Either 'same' or 'valid'
             
        Output
        -------
        Output tensor
        '''
        
        super(RATUNet3D, self).__init__()
        
        if multiplier not in [1, 2]: raise ValueError('value of multiplier can only be 1 or 2')
        
        self.dropout = nn.Dropout3d(p=dropout)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2) 
        
        # Level zero (note: suffix 'e' means encoder and 'd' means decoder)
        self.conv_e0 = conv_block(input_shape[0], base_feature, multiplier, norm, in_activation, pad)
        out_channel_encoder_0 = base_feature * multiplier #64 if multiplier=2

        # Level one 
        in_channel_encoder_1 = base_feature * 2 #64
        self.conv_e1 = conv_block(in_channel_encoder_1, in_channel_encoder_1, multiplier, norm, in_activation, pad)
        out_channel_encoder_1 = in_channel_encoder_1 * multiplier #128 if multiplier=2
        
        # Level two
        in_channel_encoder_2 = in_channel_encoder_1 * 2 #128
        self.conv_e2 = conv_block(in_channel_encoder_2, in_channel_encoder_2, multiplier, norm, in_activation, pad)
        out_channel_encoder_2 = in_channel_encoder_2 * multiplier #256 if multiplier=2
        
        # Level three
        in_channel_encoder_3 = in_channel_encoder_2 * 2 #256
        self.conv_e3 = conv_block(in_channel_encoder_3, in_channel_encoder_3, multiplier, norm, in_activation, pad)
        out_channel_encoder_3 = in_channel_encoder_3 * multiplier #512 if multiplier=2
           
        # Level two
        L = 2        
        shape_2 = int(input_shape[1]/(2**L)) #16
        in_channel_decoder_2 = out_channel_encoder_3 + shape_2 #512+16 = 528
        self.gating_1 = gating_signal(out_channel_encoder_3, out_channel_encoder_2, activation='relu', norm='batch')  # ******
        self.attn_1c = attention_block(out_channel_encoder_2, shape=shape_2)
        self.upconv_2 = up_conv(out_channel_encoder_3, out_channel_encoder_3, kernel=2, stride=2, pad=0)      
        self.conv_d2 = conv_block(in_channel_decoder_2, out_channel_encoder_2, 1, norm, in_activation, pad) # multiplier=1                         
        out_channel_decoder_2 = out_channel_encoder_2 #256
        
        # Level one
        L = 1
        shape_1 = int(input_shape[1]/(2**L)) #32
        in_channel_decoder_1 = out_channel_decoder_2 + shape_1 #256+32 = 288
        self.gating_2 = gating_signal(out_channel_decoder_2, out_channel_encoder_1, activation='relu', norm='batch')  # ******
        self.attn_2c = attention_block(out_channel_encoder_1, shape=shape_1)
        self.upconv_1 = up_conv(out_channel_decoder_2, out_channel_decoder_2, kernel=2, stride=2, pad=0)  
        self.conv_d1 = conv_block(in_channel_decoder_1, out_channel_encoder_1, 1, norm, in_activation, pad) # multiplier=1                                  
        out_channel_decoder_1 = out_channel_encoder_1 #128
        
        # Level zero
        L = 0
        shape_0 = int(input_shape[1]/(2**L)) #64
        in_channel_decoder_0 = out_channel_decoder_1 + shape_0 #128+64 = 192
        self.gating_3 = gating_signal(out_channel_decoder_1, out_channel_encoder_0, activation='relu', norm='batch')  # ******
        self.attn_3c = attention_block(out_channel_encoder_0, shape=shape_0)
        self.upconv_0 = up_conv(out_channel_decoder_1, out_channel_decoder_1, kernel=2, stride=2, pad=0) 
        self.conv_d0 = conv_block(in_channel_decoder_0, out_channel_encoder_0, 1, norm, in_activation, pad) # multiplier=1                                  
        out_channel_decoder_0 = out_channel_encoder_0
        
        # Output
        if out_activation is None:
            self.out = nn.Conv3d(out_channel_decoder_0, out_channel, kernel_size=1, stride=1, padding=pad)
        else:
            self.out = nn.Sequential(
                nn.Conv3d(out_channel_decoder_0, out_channel, kernel_size=1, stride=1, padding=pad),
                activations(out_activation)
                )
        
    def forward(self, x):

        # x = x.cpu()
        
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
        gating_1 = self.gating_1(conv_e3)        
        attn_1 = self.attn_1c(g=gating_1, x=conv_e2)
        upconv_2 = self.upconv_2(conv_e3)
        concat_2 = center_crop_and_concat(attn_1, upconv_2)
        conv_d2 = self.conv_d2(concat_2)
        conv_d2 = self.dropout(conv_d2)
        
        # Level one
        gating_2 = self.gating_2(conv_d2)       
        attn_2 = self.attn_2c(g=gating_2, x=conv_e1)
        upconv_1 = self.upconv_1(conv_d2)
        concat_1 = center_crop_and_concat(attn_2, upconv_1)
        conv_d1 = self.conv_d1(concat_1)
        conv_d1 = self.dropout(conv_d1)
        
        # Level zero
        gating_3 = self.gating_3(conv_d1)  
        attn_3 = self.attn_3c(g=gating_3, x=conv_e0)
        upconv_0 = self.upconv_0(conv_d1)       
        concat_0 = center_crop_and_concat(attn_3, upconv_0)
        conv_d0 = self.conv_d0(concat_0)
        conv_d0 = self.dropout(conv_d0)
        
        # Output
        out = self.out(conv_d0)
        
        return(out)

    
# def weight_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.zeros_(m.bias)


# Test model    
from torchsummary import summary     
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_feature = 32
    img = torch.rand(3, 1, 64, 64, 64).to(device) # batch_size, channel, height, width
    input_feature = img.size()[1]
    
    input_shape = (1, 64, 64, 64)
    
    model = RATUNet3D(input_shape, base_feature, out_channel=2, multiplier=2, 
    norm='batch', in_activation='relu', out_activation=None, dropout=0.15, pad='same')
    
    model = model.to(device)
    
    # model.apply(weight_init)
    
    out = model(img)
    
    summary(model, input_shape)
