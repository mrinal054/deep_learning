"""
RATUNet3D - Tailed residual attention UNet3D
@author: mrinal

Weights are assigned for individual conv3d layers and conv3dtranspose layers
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
        
# Weight initialization with keras default value
def weight_init_keras_default(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
        
 
# Weight initialization with truncated normal
def weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.trunc_normal_(m.weight, std=0.1)
        m.bias.data.zero_()
        
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
