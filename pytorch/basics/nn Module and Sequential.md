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
Output:
```
Input tensor shape: torch.Size([1, 3, 4, 4])
Output tensor shape: torch.Size([1, 1, 4, 4])
Output:
 tensor([[[[0.3101, 0.4306, 0.3305, 0.6101],
          [0.4712, 0.3362, 0.5149, 0.4299],
          [0.3823, 0.4192, 0.4802, 0.6037],
          [0.4182, 0.4826, 0.6056, 0.4972]]]], grad_fn=<SigmoidBackward0>)
```

We can also create a driver class `Run` that will inherit `NNModule` and will run it.

```python
class Run(NNModule):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super().__init__(in_channels, out_channels, kernel_size)
        
        self.cnn = NNModule(in_channels, out_channels, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.cnn(x)
        
        return x   
```
Now, class `Run` to generate output.

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

runner = Run(in_channels=3, out_channels=1, kernel_size=3)

out2 = runner(x)

print('Input tensor shape:', x.shape)
print('Output tensor shape:', out2.shape)
print('Output:\n', out2)  
```
Output:
```
Input tensor shape: torch.Size([1, 3, 4, 4])
Output tensor shape: torch.Size([1, 1, 4, 4])
Output:
 tensor([[[[0.4264, 0.5663, 0.5177, 0.5958],
          [0.7160, 0.6291, 0.5461, 0.4930],
          [0.3182, 0.4594, 0.5090, 0.5513],
          [0.5058, 0.5584, 0.2919, 0.5237]]]], grad_fn=<SigmoidBackward0>)
```
#### Example: nn.Module
First, let's create a class called `NNSequential` that performs sequential operations. 

```pyhon
class NNSequential(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        
        conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1)
        
        activation = nn.Sigmoid()
        
        super().__init__(conv2d, activation)
```

Note that no `forward` method is implemented explicitly. Rather, sequential layers are passed in `super`. So, this class can be used in the following way:

```python
x = torch.randn(1, 3, 4, 4)

model = NNSequential(3, 1, 3)

output = model(x)

print(output)
```
