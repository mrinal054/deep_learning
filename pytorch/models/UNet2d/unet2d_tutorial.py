# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 00:26:14 2022

@author: mrinal
"""
import torch
import torch.nn as nn

def activations(activation: str):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()

def normalization(norm: str, n_channel):
    if norm == 'batch': return nn.BatchNorm2d(n_channel)
    elif norm == 'instance': return nn.InstanceNorm2d(n_channel)
    else: print('Wrong keyword for normalization') 

def core_block(in_channel, out_channel, norm, activation, pad=0):
    '''
    It consists of conv-activation-conv-activation
    Or,
    conv-bn-activation-conv-bn-activation
    
    Input
    ------
        in_channel: No. of input channels
        out_channel: No. of output channels
        norm: Normalization. Keyword: None, 'batch', or 'instance'
        activation: Activation function. Keyword: 'relu', 'leaky', or 'elu'
        
    Output
    -------
        block: Sequential layers
    '''
    if norm == None:
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=pad),
                  activations(activation),
                  nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=pad),
                  activations(activation),
                ]
    else:
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=pad),
                  normalization(norm, out_channel),
                  activations(activation),
                  nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=pad),
                  normalization(norm, out_channel),
                  activations(activation),
                 ]
    
    block = nn.Sequential(*layers)
    return block

def up_sampling(features, pad=0):
    return nn.ConvTranspose2d(features,
                           features // 2,
                           kernel_size=2,
                           stride=2,
                           padding=pad)    

def crop_tensor(encoder_tensor, decoder_tensor):
    '''
    Crop encoder tensor to match with decoder tensor size. Required when padding=0.
    
    Input
    ------
    encoder_tensor: Encoder tensor (source tensor)
    decoder_tensor: Decoder tensor (target tensor)
    
    Output
    ------
    encoder_tensor: Cropped encoder tensor
    '''
    encoder_size = encoder_tensor.size()[2:] # height and width only
    decoder_size = decoder_tensor.size()[2:] # height and width only
    
    return encoder_tensor[
                :,
                :,
                ((encoder_size[0] - decoder_size[0]) // 2):((encoder_size[0] + decoder_size[0]) // 2),
                ((encoder_size[1] - decoder_size[1]) // 2):((encoder_size[1] + decoder_size[1]) // 2)
                ]



class UNet(nn.Module):
    def __init__(self, features, input_shape, out_channel:int, norm, activation='relu', pad=0):
        '''
        Input
        ------
             features:  e.g. [64, 128, 256, 512, 1024]
             input_shape: batch_size x channel x height x width
             out_channel: No. of channels of final output (e.g. 2)
             norm: Normalization. Keyword: None, 'batch', or 'instance'
             activation: Activation function. Keyword: 'relu', 'leaky', or 'elu'   
             
        Output
        -------
        x: Output tensor
        '''
        super(UNet, self).__init__()
        
        self.features = features
        self.out_channel = out_channel
        self.norm = norm
        self.activation = activation
        self.pad = pad
        
        # Prepare down blocks
        '''
        For instance:
            stage 1: in_channel = 1, out_channel = 64
            stage 2: in_channel = 64, out_channel = 128
            stage 3: in_channel = 128, out_channel = 256
            stage 4: in_channel = 256, out_channel = 512
            stage 5: in_channel = 512, out_channel = 1024
        '''
        self.down_block = []
        for i in range(len(self.features)):
            if i == 0:                 
                self.down_block.append(core_block(input_shape[1], self.features[i], self.norm, self.activation, self.pad))                
            else:
                self.down_block.append(core_block(self.features[i-1], self.features[i], self.norm, self.activation, self.pad))
            
        self.down_block = nn.ModuleList(self.down_block)    
        
        print('*** Down blocks ***: \n', self.down_block)
        
        # Prepare downsampling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # Prepare up blocks
        '''
        No. of up blocks is one less than the no. of down blocks
        For instance:
            stage 1: in_channel = 1024, out_channel = 512
            stage 2: in_channel = 512, out_channel = 256
            stage 3: in_channel = 256, out_channel = 128
            stage 4: in_channel = 128, out_channel = 64
        '''
        self.up_block = nn.ModuleList([
                core_block(self.features[i], self.features[i-1], self.norm, self.activation, self.pad) 
                for i in range(1,len(self.features))])
        
        print('*** Up blocks (read bottom-up) ***: \n', self.up_block)
        
        # Prepare output layer
        self.out = nn.Conv2d(in_channels=self.features[0], 
                             out_channels=self.out_channel,
                             kernel_size=1,
                             padding=self.pad)    
            
    def forward(self, x):
        'Encoder'
        for_concat = [] # this will be required for concatenation 
        for i, module in enumerate(self.down_block):
            x = module(x)
            for_concat.append(x)
            print('======================  ENCODER STAGE ' + str(i) + '  ======================')
            print('After conv and activation   : ', x.size())
            if i < len(features) - 1: # for last down block, no need of downsampling
                x = self.max_pool(x)
                print('After downsampling          : ', x.size())
        
        for_concat = for_concat[0:-1] # removing the last element as it will not be used for concatenation
        print('\nNo. of skip connections: %2d \n' % (len(for_concat)))
        
        'Decoder'     
        # No. of decoder stages is equal to the no. of skip connections
        for i in reversed(range(len(for_concat))): # e.g. i = 3,2,1,0
            print('======================  DECODER STAGE ' + str(i) + '  ======================')
            print('Decoder size before upsample: ', x.size())
            # Upsampling
            self.up_sampling = up_sampling(self.features[i+1], self.pad)
            x = self.up_sampling(x) # when i=3, then upsampling_idx=0 
            if self.norm is not None:
                self.do_norm = normalization(self.norm, self.features[i+1] // 2)
                x = self.do_norm(x)
            if self.activation is not None:
                self.do_activation = activations(self.activation)
                x = self.do_activation(x)
            # Crop encoder
            print('Decoder size                : ', x.size())
            print('Encoder size                : ', for_concat[i].size())            
            cropped_encoder = crop_tensor(for_concat[i], x) 
            print('Cropped encoder size        : ', cropped_encoder.size())
            # Concatenate
            x = torch.cat([x, cropped_encoder], 1)
            print('Concat size                 : ', x.size())
            # Conv (bn) and activation
            x = self.up_block[i](x)
            print('Final tensor                : ', x.size())
        print('=======================  OUTPUT STAGE  ========================')
        x = self.out(x)
        print('Output size                 : ', x.size())
        
        return x
            
         
# Test model         
if __name__ == "__main__":
    features = [64, 128, 256, 512, 1024]
    img = torch.rand(1, 1, 572, 572) # batch_size, channel, height, width
    input_shape = img.size()
    model = UNet(features, input_shape, out_channel=2, norm='batch', activation='relu', pad=0)
    
    out = model(img)


 