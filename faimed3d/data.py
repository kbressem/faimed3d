# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_datablock.ipynb (unless otherwise specified).

__all__ = ['ScaleDicom', 'ImageBlock3D', 'MaskBlock3D', 'show_batch_3d']

# Cell
# default_exp data

import SimpleITK as sitk
import re
import pathlib
import torchvision

from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *

# Cell
from .basics import *
from .augment import *

# Cell
class ScaleDicom(DisplayedTransform):
    "Transforms a TensorDicom3D volume to float and normalizes the data"
    def __init__(self, div=None, scale=True, normalize='mean', clamp=None, clamp_before_norm=False): store_attr()
    def encodes(self, x:(TensorDicom3D, TensorMask3D)):
        if isinstance(x, TensorMask3D): return x
        if self.clamp_before_norm and self.clamp is not None: x=x.clamp(self.clamp[0], self.clamp[1])
        if self.normalize is not None:
            if 'mean' in self.normalize: x=x.normalize('mean')
            if 'median' in self.normalize: x=x.normalize('median')
            if 'max' in self.normalize: x=x.normalize('max')
        if self.scale: x=x.hist_scaled()
        if self.div is not None: x=x/self.div
        if not self.clamp_before_norm and self.clamp is not None: x=x.clamp(self.clamp[0], self.clamp[1])
        return x.float()

# Cell
def ImageBlock3D(cls=TensorDicom3D, div=None,scale=True, normalize='mean', clamp=None, clamp_before_norm=False):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=[ScaleDicom(div=div,scale=scale,normalize=normalize, clamp=clamp, clamp_before_norm=clamp_before_norm)])

def MaskBlock3D(cls=TensorMask3D):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=[ScaleDicom])

# Cell
def show_batch_3d(dls, max_n=9, with_mask=False, alpha_mask=0.3, figsize = (15, 15), **kwargs):
    "Workarround, until implemented into dls as dls.show_batch_3d()"
    xb, yb = dls.one_batch()
    xb.show(figsize=figsize, **kwargs)
    if with_mask: yb.show(add_to_existing = True, alpha = alpha_mask, cmap = 'jet', figsize=figsize, **kwargs)