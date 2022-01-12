## UNET 2D

**PyTorch Implementation**

* For details, check `unet2d_tutorial.py`.

* For application, check `unet2d.py`.

### Test Run
```python
if __name__ == "__main__":
    features = [64, 128, 256, 512, 1024]
    img = torch.rand(1, 1, 572, 572) # batch_size, channel, height, width
    input_shape = img.size()
    model = UNet(features, input_shape, out_channel=2, norm='batch', activation='relu', pad=0)
    
    out = model(img)
```

Here is the network summary for the above test run -

```
*** Down blocks ***: 
 ModuleList(
  (0): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (4): Sequential(
    (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)
*** Up blocks (read bottom-up) ***: 
 ModuleList(
  (0): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (1): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (2): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (3): Sequential(
    (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)
======================  ENCODER STAGE 0  ======================
After conv and activation   :  torch.Size([1, 64, 568, 568])
After downsampling          :  torch.Size([1, 64, 284, 284])
======================  ENCODER STAGE 1  ======================
After conv and activation   :  torch.Size([1, 128, 280, 280])
After downsampling          :  torch.Size([1, 128, 140, 140])
======================  ENCODER STAGE 2  ======================
After conv and activation   :  torch.Size([1, 256, 136, 136])
After downsampling          :  torch.Size([1, 256, 68, 68])
======================  ENCODER STAGE 3  ======================
After conv and activation   :  torch.Size([1, 512, 64, 64])
After downsampling          :  torch.Size([1, 512, 32, 32])
======================  ENCODER STAGE 4  ======================
After conv and activation   :  torch.Size([1, 1024, 28, 28])

No. of skip connections:  4 

======================  DECODER STAGE 3  ======================
Decoder size before upsample:  torch.Size([1, 1024, 28, 28])
Decoder size                :  torch.Size([1, 512, 56, 56])
Encoder size                :  torch.Size([1, 512, 64, 64])
Cropped encoder size        :  torch.Size([1, 512, 56, 56])
Concat size                 :  torch.Size([1, 1024, 56, 56])
Final tensor                :  torch.Size([1, 512, 52, 52])
======================  DECODER STAGE 2  ======================
Decoder size before upsample:  torch.Size([1, 512, 52, 52])
Decoder size                :  torch.Size([1, 256, 104, 104])
Encoder size                :  torch.Size([1, 256, 136, 136])
Cropped encoder size        :  torch.Size([1, 256, 104, 104])
Concat size                 :  torch.Size([1, 512, 104, 104])
Final tensor                :  torch.Size([1, 256, 100, 100])
======================  DECODER STAGE 1  ======================
Decoder size before upsample:  torch.Size([1, 256, 100, 100])
Decoder size                :  torch.Size([1, 128, 200, 200])
Encoder size                :  torch.Size([1, 128, 280, 280])
Cropped encoder size        :  torch.Size([1, 128, 200, 200])
Concat size                 :  torch.Size([1, 256, 200, 200])
Final tensor                :  torch.Size([1, 128, 196, 196])
======================  DECODER STAGE 0  ======================
Decoder size before upsample:  torch.Size([1, 128, 196, 196])
Decoder size                :  torch.Size([1, 64, 392, 392])
Encoder size                :  torch.Size([1, 64, 568, 568])
Cropped encoder size        :  torch.Size([1, 64, 392, 392])
Concat size                 :  torch.Size([1, 128, 392, 392])
Final tensor                :  torch.Size([1, 64, 388, 388])
=======================  OUTPUT STAGE  ========================
Output size                 :  torch.Size([1, 2, 388, 388])
```


### Reference
* Original paper: [here](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28).