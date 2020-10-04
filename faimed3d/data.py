# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03-custom-3d-datablock.ipynb (unless otherwise specified).

__all__ = ['ScaleDicom', 'ImageBlock3D', 'AddMaskCodes3D', 'MaskBlock3D', 'show_batch_3d']

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
    def __init__(self, div=None, div_mask=1): store_attr()

    def encodes(self, x:(TensorDicom3D, TensorMask3D)):

        if isinstance(x, TensorMask3D): return x

        if self.div is None:
            return normalize(x.hist_scaled()).float()
        else:
            return (x.hist_scaled()/div).float()


# Cell
def ImageBlock3D(cls=TensorDicom3D):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=ScaleDicom)

# Cell
class AddMaskCodes3D(AddMaskCodes):
    "Add the code metadata to a `TensorMask`"

    def decodes(self, o:TensorMask3D):
        o = o.int()
        if self.codes is not None: o._meta = {'codes': self.codes}
        return o

def MaskBlock3D(codes = None):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=TensorMask3D.create, item_tfms=AddMaskCodes3D(codes=codes),  batch_tfms=ScaleDicom)


# Cell
def show_batch_3d(dls, max_n=9, with_mask=False, alpha_mask=0.3, figsize = (15, 15), **kwargs):
    "Workarround, until implemented into dls as dls.show_batch_3d()"

    xb, yb = dls.one_batch()

    if xb.device.type == 'cuda':
        xb=xb.cpu()
        yb=yb.cpu()
    if xb.ndim < 4: raise TypeError('Batch is not a batch of multiple 3D images')
    if xb.ndim == 5:
        print('Expected 4D tensor but got 5D tensor. Removing the last dimension under the assumption that it is a color channel ')
        xb = xb[:,:,:,:,0]
        yb = yb[:,:,:,:,0]
    if xb.ndim > 5: raise NotImplementedError('Batches with more than 3 Dimensions are currently not supported')

    if with_mask:

        x_grid = show_multiple_3d_images(xb, return_grid = True, **kwargs)[0,:,:]
        y_grid = show_multiple_3d_images(yb, return_grid = True, **kwargs)[0,:,:]

        plt.figure(figsize=figsize)
        plt.imshow(x_grid, cmap = 'gray')
        plt.imshow(y_grid, cmap = 'jet', alpha = alpha_mask)

    else: show_multiple_3d_images(xb, **kwargs)
