# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_dataloaders.ipynb (unless otherwise specified).

__all__ = ['PreprocessDicom', 'AddColorChannel', 'ImageBlock3D', 'MaskBlock3D', 'ImageDataLoaders3D',
           'SegmentationDataLoaders3D']

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
from .preprocess import *
from .augment import *

# Cell
class PreprocessDicom(DisplayedTransform):
    "Transforms a TensorDicom3D volume to float and normalizes the data"
    def __init__(self, rescale=True,
                 correct_spacing=True, spacing = 1,
                 correct_orientation=False, orientation=(0,0,0,0,0,0),
                 clip_high_values=False, clipping_range=(-2000, 10000)):
        """
        Args:
            rescale (bool): if pixel values should be scaled according to the rescale intercept and slope
            correct_spacing (bool): if spacing of all volumes should be set to a unified value.
            spacing (int): new pixel spacing, default is 1.
            correct_orientation (bool): if orientation of volumes should be the same across all items.
            orientation (tuple): new porientation.
            clip_high_values (bool): if abriatry high values shoule be clipped
            clipping_rang (tuple or list): min and maximum to clip values
        """

        store_attr()
    def encodes(self, x:(TensorDicom3D, TensorMask3D)):
        if isinstance(x, TensorMask3D): return x
        if hasattr(x, 'metadata'): # numpy arrays will not have metadata
            if self.rescale: x = x.rescale_pixeldata()
            if self.correct_spacing: x = x.size_correction(self.spacing)
            if self.correct_orientation: raise NotImplementedError()
        if self.clip_high_values: x = x.clamp(self.clipping_range[0],
                                              self.clipping_range[1])
        return x.float()

# Cell
class AddColorChannel(RandTransform):
    "Transforms a TensorDicom3D volume to float and normalizes the data"
    def __init__(self, p=1.):
        store_attr()
    def encodes(self, x:(TensorDicom3D, TensorMask3D)):
        if x.ndim == 3: return torch.stack((x, )*3, 0)
        elif x.ndim == 4: return torch.stack((x, )*3, 1)
        else: raise ValueError('expeted Tensor with 3 or four dimensions but got {}'.format(x.ndim))

# Cell
def ImageBlock3D(cls=TensorDicom3D, **kwargs):
    "A `TransformBlock` for images of `cls`. For possible kwargs see PreporcessDicom."
    return TransformBlock(type_tfms=cls.create, batch_tfms=[PreprocessDicom(**kwargs)])

def MaskBlock3D(cls=TensorMask3D, codes=None):
    "A `TransformBlock` for images of `cls`. For possible kwargs see PreporcessDicom."
    return TransformBlock(type_tfms=cls.create, item_tfms=AddMaskCodes(codes=codes))

# Cell
class ImageDataLoaders3D(DataLoaders):
    "Enarly identitcal to fastai `ImageDataLoaders` but adapted for 3D data with some preprocessing steps added"

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, rescale_method=None, clean_tmpdir=True,**kwargs):
        "Create from `df` using `fn_col` and `label_col`"

        if clean_tmpdir:
            print('Cleaning tmpdir.')
            cleanup_tmpdir() # clean up tmpdir from last session
            print('You can disable automatic cleanup of the tmpdir (e.g. when doing multiple sessions in parallel)'
                  ' with setting clean_tmpdir=False')
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        if rescale_method is None:
            print('No rescale method was used. This is not advisable due to high risk of exploding gradients. '
                  'Falling back to mean scaling.')
            rescale_method = MeanScale()
        if item_tfms is not None:
            item_tfms = [rescale_method, *item_tfms] if isinstance(item_tfms, list) else [rescale_method, item_tfms]
        else: item_tfms = rescale_method

        if batch_tfms is not None:
            batch_tfms = [*batch_tfms, AddColorChannel()] if isinstance(batch_tfms, list) else [batch_tfms, AddColorChannel()]
        else: batch_tfms = AddColorChannel()

        dblock = DataBlock(blocks=(ImageBlock3D(cls=TensorDicom3D), y_block),
                           get_x=ColReader(fn_col, pref=pref, suff=suff),
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, df, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, **kwargs):
        "Create from `path/csv_fname` using `fn_col` and `label_col`"
        df = pd.read_csv(Path(path)/csv_fname, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_folder(cls, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, item_tfms=None,
                    batch_tfms=None, **kwargs):
        "Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)"
        raise NotImplementedError('Currently only from_df and from_csv are supported. '
                                  'You can raise a feature request on Github (https://github.com/kbressem/faimed3d/), if you need this feature implemented. ')

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_path_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`"
        raise NotImplementedError('Currently only from_df and from_csv are supported. '
                                  'You can raise a feature request on Github (https://github.com/kbressem/faimed3d/), if you need this feature implemented. ')

    @classmethod
    def from_name_func(cls, path, fnames, label_func, **kwargs):
        "Create from the name attrs of `fnames` in `path`s with `label_func`"
        f = using_attr(label_func, 'name')
        return cls.from_path_func(path, fnames, f, **kwargs)

    @classmethod
    def from_path_re(cls, path, fnames, pat, **kwargs):
        "Create from list of `fnames` in `path`s with re expression `pat`"
        return cls.from_path_func(path, fnames, RegexLabeller(pat), **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_name_re(cls, path, fnames, pat, **kwargs):
        "Create from the name attrs of `fnames` in `path`s with re expression `pat`"
        return cls.from_name_func(path, fnames, RegexLabeller(pat), **kwargs)


# Cell
@patch
def show_batch_3d(dls:DataLoaders, max_n=9, with_mask=False, alpha_mask=0.3, figsize = (15, 15), **kwargs):
    "Workarround, until implemented into dls as dls.show_batch_3d()"
    xb, yb = dls.one_batch()
    xb.show(figsize=figsize, **kwargs)
    if with_mask: yb.show(add_to_existing = True, alpha = alpha_mask, cmap = 'jet', figsize=figsize, **kwargs)

# Cell
class SegmentationDataLoaders3D(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for segmentation problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)

    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, codes=None,
                valid_col=None, item_tfms=None, batch_tfms=None, rescale_method=None, clean_tmpdir=True,**kwargs):
        "Create from `df` using `fn_col` and `label_col`"

        if clean_tmpdir:
            print('Cleaning tmpdir.')
            cleanup_tmpdir() # clean up tmpdir from last session
            print('You can disable automatic cleanup of the tmpdir (e.g. when doing multiple sessions in parallel)'
                  ' with setting clean_tmpdir=False')
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        if rescale_method is None:
            print('No rescale method was used. This is not advisable, due to high risk of exploding gradients. '
                  'Falling back to mean scaling.')
            rescale_method = MeanScale()
        if item_tfms is not None:
            item_tfms = [rescale_method, *item_tfms] if isinstance(item_tfms, list) else [rescale_method, item_tfms]
        else: item_tfms = rescale_method

        if batch_tfms is not None:
            batch_tfms = [*batch_tfms, AddColorChannel()] if isinstance(batch_tfms, list) else [batch_tfms, AddColorChannel()]
        else: batch_tfms = AddColorChannel()

        dblock = DataBlock(blocks=(ImageBlock3D(cls=TensorDicom3D), MaskBlock3D(codes=codes)),
                           get_x=ColReader(fn_col, pref=pref, suff=suff),
                           get_y=ColReader(label_col, pref=pref, suff=suff),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, df, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, **kwargs):
        "Create from `path/csv_fname` using `fn_col` and `label_col`"
        df = pd.read_csv(Path(path)/csv_fname, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)