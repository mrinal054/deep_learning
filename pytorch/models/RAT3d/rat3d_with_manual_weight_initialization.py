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
      
