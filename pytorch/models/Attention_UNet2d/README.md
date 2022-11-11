## Attention UNET 2D

**PyTorch implementation**

### Test Run

```python
# Test model         
if __name__ == "__main__":
    features = [64, 128, 256, 512, 1024]
    img = torch.rand(1, 1, 572, 572) # batch_size, channel, height, width
    input_shape = img.size()
    model = UNet(features, input_shape, out_channel=2, norm='batch', activation='relu', pad=0,
                 att_activation='relu', att_norm=None)
    
    out = model(img)
```

* Choose `activation` or `att_activation` functions from: `relu`, `leaky`, `elu`.
* Choose `norm` or `att_norm` from: `None`, `batch`, `instance`.

### Reference:
* Original paper: [here](https://arxiv.org/abs/1804.03999).