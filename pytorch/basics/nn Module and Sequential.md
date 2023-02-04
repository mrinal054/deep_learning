## nn.Module() vs nn.Sequential()
* `nn.Module` is the base class to implement any neural network in PyTorch, whether it is sequential or not.
* It requires a `forward` method to execute.
* `nn.Sequntial` is a subclass of `nn.Module` and can be used when layers are sequential.
* One advantage of `nn.Sequential` is that we don't need to implement the `forward` method.


#### Example: nn.Module

Create a class called `NNModule`.

```python
import torch
import torch.nn as nn

class NNModule(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size=3):        
        super(head, self).__init__()        
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1)        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):        
        x = self.conv2d(x)        
        x = self.activation(x)
        
        return x
```
Now, run this class.

```python
# Set seed        
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False        

# Create a torch with size (batch, C, H, W)        
x = torch.randn(1, 3, 4, 4)        
    
# Run model
model = NNModule(in_ch=3, out_ch=1, kernel_size=3)

out = model(x)

print('Input tensor shape:', x.shape)
print('Output tensor shape:', out.shape)
print('Output:\n', out) 
```
