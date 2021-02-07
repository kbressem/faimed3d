# FAIMED 3D
> fastai extension for medical 3d images including 3d transforms, datablocks and novel network architectures. 


## Install

`pip install faimed3d`

In contrast to fastai, which uses Pydicom to read medical images, faimed3d uses SimpleITK, as it supports more image formats.  
Currently faimed3d is build using the following versions of fastai, fastcore, nbdev, PyTorch, torchvision and SimpleITK

```python
import fastai
import fastcore
import nbdev
import torch
import torchvision

print('fastai:', fastai.__version__)
print('fastcore:', fastcore.__version__)
print('nbdev:', nbdev.__version__)
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('SimpleITK: 2.0.0rc3 (ITK 5.1)')
```

    fastai: 2.1.4
    fastcore: 1.3.2
    nbdev: 0.2.40
    torch: 1.7.1
    torchvision: 0.8.2
    SimpleITK: 2.0.0rc3 (ITK 5.1)


## Example 3D Classification

```python
from faimed3d.all import *
from torchvision.models.video import r3d_18
```

```python
d = pd.read_csv('../data/radiopaedia_cases.csv')
```

Defining paramteres for piecewise histogram scaling. Paramters can be obtained from dataloaders running `dls.standard_scale_from_dls()`

`faimed3d` keeps track of the metadata until the items are concatenated as a batch. 

```python
dls = ImageDataLoaders3D.from_df(d,
                                 item_tfms = ResizeCrop3D(crop_by = (0, 6, 6), resize_to = (20, 112, 112)),
                                 bs = 2, val_bs = 2)
```

Construct a learner similar to fastai, even transfer learning is possible using the pretrained resnet18 from torchvision.

```python
learn = cnn_learner_3d(dls, r3d_18, pretrained=False) 
learn = learn.to_fp16()
```

```python
#slow
learn.lr_find()
```








    SuggestedLRs(lr_min=6.918309954926372e-05, lr_steep=1.5848931980144698e-06)




![png](docs/images/output_12_2.png)

