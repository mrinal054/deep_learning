## nn.Module() vs nn.Sequential()
* `nn.Module` is the base class to implement any neural network in PyTorch, whether it is sequential or not.
* It requires a `forward` method to execute.
* `nn.Sequntial` is a subclass of `nn.Module` and can be used when layers are sequential.
* One advantage of `nn.Sequential` is that we don't need to implement the `forward` method.


#### Example: nn.Module

```python
import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        
        super(head, self).__init__()
        
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.conv2d(x)
        
        x = self.activation(x)
        
        return x
```
