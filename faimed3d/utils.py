# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06-various-tools.ipynb (unless otherwise specified).

__all__ = ['RotateNifti', 'SubvolumeExporter', 'CropOriginalToMask']

# Cell
# default_exp utils

from fastai.basics import *
import pathlib
import SimpleITK as sitk

# Cell

from .basics import *
from .augment import *
from .data import *
from .models import *

# Cell
class RotateNifti():
    def __init__(self):
        warn('This class will change the source files on you disk. Be carefull.')

    def rotate_single(self, fn:(Path, str)):
        if isinstance(fn, Path): fn = str(fn)

        im = sitk.ReadImage(fn)
        arr = sitk.GetArrayFromImage(im)
        arr = np.rot90(arr, 0)
        arr = np.flip(arr, 1)
        im2 = sitk.GetImageFromArray(arr)
        for k in im.GetMetaDataKeys(): # Copy meta data from original image before overwriting it.
            im2.SetMetaData(k, im.GetMetaData(k))
        sitk.WriteImage(im2, fn)

    def rotate_list(self, file_list=None, verbose = True):
        if file_list is not None: self.file_list = file_list

        for fn in file_list:
            self.rotate_single(fn)
            if verbose: print('converted file at: '+str(fn))

# Cell
class SubvolumeExporter(object):
    def __init__(self):
        self.model = None
        self.item_tfms = None

    def assign_model(self, model):
        self.model = model

    def assign_tfms(self, tfms):
        "assign transformations which should be applied to the surce, before size of source and mask are matched"
        self.item_tfms = tfms

    def merge_mask(self, mask):
        mask = mask.squeeze()
        if mask.ndim == 4:
            mult_channel_by = tensor(range(0, mask.size(0)))
            mask = mask * mult_channel_by[:, None, None, None]
            return torch.sum(mask, 0)
        else: return mask

    def match_size_mask_source(self):
        "rescales the segmentation mask to the original image resolution"

        if self.item_tfms == None: raise TypeError('No item_tfms specified.')
        source = TensorDicom3D.create(self.item_path)
        self.metadata = source.metadata
        source = self.item_tfms(source) # will lose metadata in transforms

        x,y = source.size()[1:]
        z = self.mask.size(0)
        source = source.resize_3d(size = (z,x,y), mode = 'trilinear')
        self.mask = self.mask.resize_3d(size = source.size(), mode = 'nearest')
        self.mask = TensorMask3D(self.mask)
        self.orig = TensorMask3D(source)
        self.mask.metadata = self.metadata
        self.orig.metadata = self.metadata


    def predict(self, item, rm_type_tfms=None):
        self.item_path = item
        _, self.mask, _ = self.model.predict(self.item_path, rm_type_tfms)
        self.mask = self.mask.round()
        self.mask = self.merge_mask(self.mask)
        self.mask = TensorMask3D(self.mask)

        self.match_size_mask_source()

    def show_pair(self, alpha = 0.25, **kwargs):
        self.orig.show(**kwargs)
        self.mask.show(add_to_existing = True, alpha = alpha, cmap = 'jet', **kwargs)

    def strip_pair(self, pad_z = 1, pad_xy = 5):
        "padds the idexes, so that a small margin of zeros remains"

        self.orig.strip_idx = self.mask.get_strip_idx(symmetric=True)
        self.mask = self.mask.strip(pad_z = pad_z, pad_xy=pad_xy)
        self.orig = self.orig.strip(pad_z = pad_z, pad_xy=pad_xy)
        self.orig.metadata = self.metadata
        self.mask.metadata = self.metadata

    def convert_and_export(self, source, orig_name, mask_name, pad_z, pad_xy, verbose):
        """
        Reads a list of source images and uses a given learner to predict the mask. Crops the mask and source image an then exports the files.

        Args:
            source: str or path. The path to the original image. Can be a DICOM direcetory or a single DICOM, NIfTI, NRRD, Analyze file (any type supported by SimpleITK)
            orig_name: str or path. New filename for the cropped original image
            mask_name: str or path. New filename for the predicted mask.

        returns:
            Nothing. Writes files to disk.
        """
        if self.model is None: raise NameError('No model for predictions assigned. Assing a model to the {} using {}.assign_model(model)'.format(self.__class__.__name__, self.__class__.__name__))

        self.predict(source)
        self.strip_pair(pad_z = pad_z, pad_xy = pad_xy)
        self.orig.save(orig_name)
        self.mask.save(mask_name)

        if verbose: print('wrote image to {} and mask to {}'.format(str(orig_name), str(mask_name)))

    def convert_and_export_list(self, source, orig_name, mask_name, pad_z = 1, pad_xy = 5, verbose = True):
        if not isinstance(source, list) or not isinstance(orig_name, list) or not isinstance(mask_name, list): raise TypeError('source, orig_name, mask_name need to be lists of equal size')
        if len(source) != len(orig_name) or len(source) != len(mask_name): raise TypeError('source, orig_name, mask_name need to be lists of equal size')

        for s, o, m, in zip(source, orig_name, mask_name):
            self.convert_and_export(s, o, m, pad_z, pad_xy, verbose)

# Cell
class CropOriginalToMask(SubvolumeExporter):

    def load_pair(self, image, mask):
        self.orig = TensorMask3D.create(image)
        self.mask = TensorMask3D.create(mask)
        self.metadata = self.orig.metadata

        self.mask.metadata = self.metadata

    def export_pair(self, new_image, new_mask, pad_z, pad_xy):
        self.strip_pair(pad_z = pad_z, pad_xy = pad_xy)
        self.orig.save(new_image)
        self.mask.save(new_mask)

    def convert_and_export(self, image, mask, new_image=None, new_mask=None, pad_z=1, pad_xy=10, verbose = True):
        if isinstance(image, str): image = Path(image)
        if isinstance(mask, str): mask = Path(mask)

        if new_image is None:
            if image.is_dir():
                new_image = image/'cropped_volume.nii.gz'
            else:
                new_image = image.parent/'cropped_volume.nii.gz'


        if new_mask is None:
            if mask.is_dir():
                new_mask = mask/'cropped_mask.nii.gz'
            else:
                new_mask = mask.parent/'cropped_mask.nii.gz'

        self.load_pair(image, mask)
        self.export_pair(new_image, new_mask, pad_z, pad_xy)

        if verbose: print('wrote image to {} and mask to {}'.format(str(new_image), str(new_mask)))

    def convert_and_export_list(self, image, mask, new_image=None, new_mask=None,  pad_z = 1, pad_xy = 10, verbose = True):
        if not isinstance(image, list) or not isinstance(mask, list): raise TypeError('source, orig_name, mask_name need to be lists of equal size')

        if len(image) != len(mask): raise TypeError('source, orig_name, mask_name need to be lists of equal size')

        if new_image is None: new_image = [None for i in image]
        if new_mask is None: new_mask = [None for i in mask]


        for i, m, ni, nm in zip(image, mask, new_image, new_mask):
            self.convert_and_export(image=i, mask=m, new_image=ni, new_mask=nm, pad_z=pad_z, pad_xy=pad_xy, verbose=verbose)