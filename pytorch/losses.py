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
        
        Input
        --------
        weights: (tuple or list) weight for each class. e.g. [1, 150.] when num_classes=2
        
        Output
        --------
        Returns weighted cross-entropy loss    
        
        Expanation
        --------------        
        weights = [1, 100, 150, 1.]
        
        # Get dimensions to reshape the weights
        # E.g. if weights = [1, 100, 150, 1.], then reshape dims = [1, 4, 1, 1, 1] 
        # Here 4 is no. of channels/classes or len(weights). All are same.
        # 1s are for batch, height, width, and depth
        
        num_classes = len(weights)
        
        reshape_dims = [1] + [num_classes] + [1,1,1]
        
        weights = torch.tensor(weights) # dtype=torch.float32
        
        reshaped_weights = weights.view(reshape_dims) # for num_classes=4, size is now torch.Size([1, 4, 1, 1, 1])
        
        y_true = torch.randn(3, 4, 64, 64, 64)
        y_pred = torch.randn(3, 4, 64, 64, 64)
        
        y_true = y_true.type(torch.float32)
        y_pred = y_pred.type(torch.float32)
        
        weighted_y_true = torch.sum(reshaped_weights*y_true, dim = 1) 
        
        loss = nn.CrossEntropyLoss(reduction='none')
        
        ce_loss = loss(y_pred, y_true)
        
        wce_loss = ce_loss * weighted_y_true
        
        mean_wce_loss = torch.mean(wce_loss)
        
        Example
        ------------        
        y_true = torch.randn(3, 4, 64, 64, 64)
        y_pred = torch.randn(3, 4, 64, 64, 64)

        y_true = y_true.type(torch.float32)
        y_pred = y_pred.type(torch.float32)

        loss = WCELoss([1, 100, 150, 1.])            

        wce_loss = loss(y_pred, y_true)
        
        '''
