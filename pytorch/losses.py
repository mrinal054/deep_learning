"""
Created on Sat Jul 30 11:51:42 2022

@author: mrinal
"""
import torch.nn as nn
import torch 

class WCELoss(nn.Module):

    def __init__(self, weights):

        super(WCELoss, self).__init__()
        
        '''
        It calculates weighted cross-entropy loss.
        
        Warning
        --------
        Do not use softmax in the final layer. WCELoss utilizes pytorch's nn.CrossEntropyLoss(),
        which applies softmax before calculating the loss. So, a softmax actiation in
        the final layer will result in an incorrect loss value.  
        
