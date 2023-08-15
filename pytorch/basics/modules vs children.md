## modules() vs children()

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.fc = nn.Linear(64, 10)
        self.double_conv = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=3),
                                nn.Conv2d(3, 64, kernel_size=3)])
```

**model.modules():**
* This function recursively goes through all the modules in the model, including the model itself, its sub-modules, and their sub-modules, forming a flattened structure. <br>
* It returns an iterator over all the modules, including the model itself and all its sub-modules in a depth-first manner. <br>

Let's print modules for the above model.
```python
model = MyModel()

for module in model.modules():
    print(module)
```
Output:
```
MyModel(
  (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
  (fc): Linear(in_features=64, out_features=10, bias=True)
  (double_conv): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
  )
)
Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
Linear(in_features=64, out_features=10, bias=True)
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
  (1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
)
Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
```
As you can see, it returns **modules as well as the model itself**. 
