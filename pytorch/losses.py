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
        
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
        self.num_classes = len(weights)
        
        self.weights = torch.tensor(weights) # dtype=torch.float32
        

    def forward(self, y_pred, y_true, device: str=None):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        y_pred = y_pred.type(torch.float32).to(device) # Cast as float32
        
        y_true = y_true.type(torch.float32).to(device) # Cast as float32

        # Get dimensions to reshape the weights
        # E.g. if weights = [1, 100, 150, 1.], then reshape dims = [1, 4, 1, 1, 1] (for 3d)
        # Here 4 is no. of channels/classes or len(weights). All are same.
        # 1s are for batch, height, width, and depth
        
        if y_true.dim() == 5: reshape_dims = [1] + [self.num_classes] + [1,1,1] # for 3d
        
        elif y_true.dim() == 4: reshape_dims = [1] + [self.num_classes] + [1,1] # for 2d
        
        else: raise Exception(f'Incorrect dimension. A tensor should have 4 (for 2d) or 5 (for 3d) dimensions. Instead, has {y_true.dim()} dimensions.')
        
        reshaped_weights = self.weights.view(reshape_dims) # for num_classes=4, size is now torch.Size([1, 4, 1, 1, 1])
        
        weighted_y_true = torch.sum(reshaped_weights.to(device) * y_true, dim = 1) 
        
        ce_loss = self.loss(y_pred, y_true)
        
        wce_loss = ce_loss * weighted_y_true
        
        return torch.mean(wce_loss).to(device) # return the mean value
        
