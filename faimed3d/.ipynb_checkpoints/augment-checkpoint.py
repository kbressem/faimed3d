# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02-transformation-for-3d-data.ipynb (unless otherwise specified).

__all__ = ['Resize3D', 'RandomFlip3D', 'RandomRotate3D', 'RandomRotate3DBy', 'RandomDihedral3D', 'RandomCrop3D',
           'ResizeCrop3D', 'RandomWarp3D', 'RandomNoise3D', 'RandomBrightness3D', 'RandomContrast3D']

# Cell
# default_exp augment
import torchvision
import torch
import fastai
from fastai.basics import *
from fastai.vision.augment import *

# Cell
from .basics import *

# Cell

@patch
def resize_3d(t: (TensorDicom3D), size: int):

    '''
    A function to resize a 3D image using torch.nn.functional.grid_sample

    Taken form the offical documention:
        Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.
        In the spatial (4-D) case, for input with shape (N,C,Hin,Win) and with grid in shape (N, Hout, Wout, 2), the output will have shape (N, C, Hout,Wout)

        In the case of 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w].
        mode argument specifies nearest or bilinear interpolation method to sample the input pixels.

    Workflow of this function:
    1. create a fake RGB 3D image through generating fake color channels.
    2. add a 5th batch dimension
    3. create a flow-field for rescaling:
        a. create a 1D tensor giving a linear progression from -1 to 1
        b. creat a mesh-grid (the flow field) from x,y,z tensors from (a)
    4. resample the input tensor according to the flow field
    5. remove fake color channels and batch dim, returning only the 3D tensor

    Args:
        t (Tensor): a Rank 3 Tensor to be resized
        new_dim (int): a tuple with the new x,y,z dimensions of the tensor after resize

    '''
    if type(size) in (tuple, fastuple) and len(size) == 3:
        z,x,y = size # for a reason, I do currently not understand, order of the axis changes from resampling. flipping the order of x,y,z is the current workaround
    else:
        raise ValueError('"size" must be a tuple with length 3, specifying the new (x,y,z) dimensions of the 3D tensor')

    t = torch.stack((t,t,t)) # create fake color channel
    t = t.unsqueeze(0).float() # create batch dim

    x = torch.linspace(-1, 1, x) # create resampling 'directions' for pixels in each axis
    y = torch.linspace(-1, 1, y)
    z = torch.linspace(-1, 1, z)

    meshx, meshy, meshz = torch.meshgrid((x, y, z)) #
    grid = torch.stack((meshy, meshx , meshz), 3) # create flow field. x and y need to be switched as otherwise the images are rotated.
    grid = grid.unsqueeze(0) # add batch dim
    t_resized = F.grid_sample(t, grid, align_corners=True, mode = 'bilinear') # rescale the 5D tensor
    return t_resized[0,0,:,:,:].permute(2,0,1).contiguous() # remove fake color channels and batch dim, reorder the image (the Z axis has moved to the back...)

class Resize3D(RandTransform):
    split_idx,order = None, 1
    "Resize a 3D image"

    def __init__(self, size, **kwargs):
        size = _process_sz_3d(size)
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: TensorDicom3D): return x.resize_3d(self.size)

def _process_sz_3d(size):
    if len(size) == 2: size=(size[0],size[1], size[1])
    return fastuple(size[0],size[1],size[2])

# Cell

@patch
def flip_ll_3d(t: TensorDicom3D):
    "flips an image laterolateral"
    return t.flip(-1)

@patch
def flip_ap_3d(t: TensorDicom3D):
    "flips an image anterior posterior"
    return t.flip(-2)

@patch
def flip_cc_3d(t: TensorDicom3D):
    "flips an image cranio caudal"
    return t.flip(-3)


# Cell

class RandomFlip3D(RandTransform):
    "Randomly flip alongside any axis with probability `p`"
    def __init__(self, p=0.75):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        "Set `self.do` based on `self.p`"
        self.do = self.p==1. or random.random() < self.p
        self.axis = random.randint(1, 3)*-1  # add a random integer for axis to rotate

    def encodes(self, x:TensorDicom3D):
        return x.flip(self.axis)

# Cell

@patch
def rotate_90_3d(t: TensorDicom3D):
    return t.transpose(1, 2)

@patch
def rotate_270_3d(t: TensorDicom3D):
    return t.transpose(1, 2).flip(-1)

@patch
def rotate_180_3d(t: TensorDicom3D):
    return t.flip(-1).flip(-2)


class RandomRotate3D(RandTransform):
    "Randomly flip rotates the axial slices of the 3D image 90/180 or 270 degrees with probability `p`"
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        "Set `self.do` based on `self.p`"
        self.do = self.p==1. or random.random() < self.p
        self.which = random.randint(1, 3)  # add a random integer for axis to rotate

    def encodes(self, x:TensorDicom3D):
        if self.which == 1: return x.rotate_90_3d()
        elif self.which == 2: return x.rotate_180_3d()
        else: return x.rotate_270_3d()


# Cell

@patch
def rotate_3d_by(t: TensorDicom3D, angle: (int, float), axes: list):
    '''rotates 2D slices of a 3D tensor.
    Args:
        t: a TensorDicom3D object or torch.tensor
        angle: the angle to rotate the image
        axes: axes to which the rotation should be applied.

    Example:
        If the tensor `t` has the shape of (10, 512, 512), which is equal to 10 slices of 512x512 px.

    rotate_3d_by(t, angle = -15.23, axes = [1,2]) will rotate each slice for -15.23 degrees.

    '''
    rot_t = torch.from_numpy(ndimage.rotate(t, angle, axes, reshape=False))
    return retain_type(rot_t, typ = TensorDicom3D)


class RandomRotate3DBy(RandTransform):
    "Randomly flip rotates the axial slices of the 3D image 90/180 or 270 degrees with probability `p`"
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        "Set `self.do` based on `self.p`"
        self.do = self.p==1. or random.random() < self.p
        self.angle = random.randint(-10, 10)  # add a random integer for axis to rotate
        self.axes = random.choice([[0,1],[1,2],[0,2]])

    def encodes(self, x:TensorDicom3D):
        return x.rotate_3d_by(angle=self.angle, axes=self.axes)

# Cell

@patch
def dihedral3d(x:TensorDicom3D, k):
    "apply dihedral transforamtions to the 3D Dicom Tensor"
    if k in [6,7,8,9,14,15,16,17]: x = x.flip(-3)
    if k in [4,10,11,14,15]: x = x.flip(-1)
    if k in [5,12,13,16,17]: x = x.flip(-2)
    if k in [1,7,10,12,14,16]: x = x.transpose(1, 2)
    if k in [2,8]: x = x.flip(-1).flip(-2)
    if k in [3,11,13,15,17]: x = x.transpose(1, 2).flip(-1)
    return x

class RandomDihedral3D(RandTransform):
    "randomly flip and rotate the 3D Dicom volume with a probability of `p`"
    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.k = random.randint(0,17)

    def encodes(self, x:TensorDicom3D): return x.dihedral3d(self.k)


# Cell

@patch
def crop_3d(t: TensorDicom3D, margins: (int, float), perc_margins = False):
    "Similar to function `crop_3d_tensor`, but no checking for margin formats is done, as they were correctly passed to this function by RandomCrop3D.encodes"

    x, y, z = t.shape
    x1,x2,y1,y2,z1,z2 = margins

    if perc_margins:
        if not all(isinstance(i, float) for i in [x1,x2,y1,y2,z1,z2]): raise ValueError('percentage margins should be a float value between 0 and 0.5')
        x1,x2,y1,y2,z1,z2 = int(x1*x),int(x2*x),int(y1*y),int(y2*y),int(z1*z),int(z2*z)

    return retain_type(t[x1:x-x2, y1:y-y2, z1:z-z2], typ = TensorDicom3D)


class RandomCrop3D(RandTransform):
    '''
    Randomly crop the 3D volume with a probability of `p`
    The x axis is the "slice" axis, where no cropping should be done by default
    '''

    def __init__(self, final_margins, crop_by, perc_margins=False, p=1, **kwargs):
        super().__init__(p=p,**kwargs)
        self.p = p
        self.final_margins = final_margins
        self.perc_margins = perc_margins
        self.crop_by_x, self.crop_by_y, self.crop_by_z = crop_by

#    def setups(self, items):
#        self.final_margins, crop_by, self.perc_margins = items
#        self.crop_by_x, self.crop_by_y, self.crop_by_z = crop_by

    def encodes(self, x:TensorDicom3D):
        if self.p < 0.5: self.margins = self.final_margins
        else:
            if type(self.final_margins) is tuple and len(self.final_margins) == 3:
                cropx, cropy, cropz = self.final_margins
                try:
                    if len(cropx) == 2:
                        x1,x2 = cropx
                    if len(cropx) == 2:
                        y1,y2 = cropy
                    if len(cropx) == 2:
                        z1,z2 = cropz
                except:
                    x1,x2,y1,y2,z1,z2 = cropx, cropx, cropy, cropy, cropz, cropz

                self.x_add = random.randint(-self.crop_by_x,self.crop_by_x)
                self.y_add = random.randint(-self.crop_by_y,self.crop_by_y)
                self.z_add = random.randint(-self.crop_by_z,self.crop_by_z)

                self.margins = (x1+self.x_add, x2-self.x_add,y1+self.y_add, y2-self.y_add,z1+self.z_add, z2-self.z_add)

            else:
                raise ValueError('"final_margins" must be a tuple with length 3')

            if any(self.margins) < 0: raise ValueError('cannot crop to a negative dimension')

        return x.crop_3d(margins = self.margins, perc_margins = self.perc_margins)


# Cell

class ResizeCrop3D(RandTransform):
    split_idx,order = None, 1

    "Resize and crop a 3D tensor"

    def __init__(self, crop_by, resize_to, perc_crop=False, p=1, **kwargs):
        resize_to = _process_sz_3d(resize_to)
        crop_by = crop_by
        perc_crop = perc_crop
        store_attr()
        super().__init__(p=p,**kwargs)

    def encodes(self, x:TensorDicom3D):
        if type(self.crop_by) is tuple and len(self.crop_by) == 3:
            cropx, cropy, cropz = self.crop_by
            try: x1,x2 = cropx
            except: x1,x2 = cropx,cropx
            try: y1,y2 = cropy
            except: y1,y2 = cropy,cropy
            try: z1,z2 = cropz
            except: z1,z2 = cropz,cropz

            self.margins = (x1,x2,y1,y2,z1,z2)

        else: raise ValueError('"crop_by" must be a tuple with length 3 in the form ox (x,y,z) or ((x1,x2),(y1,y2),(z1,z2))')
        if any(self.margins) < 0: raise ValueError('cannot crop to a negative dimension')

        return x.crop_3d(margins = self.margins, perc_margins = self.perc_crop).resize_3d(self.resize_to)


# Cell

@patch
def warp_3d(t: TensorDicom3D):


    z,x,y = t.shape # for a reason, I do currently not understand, order of the axis changes from resampling. flipping the order of x,y,z is the current workaround. z axis is now the slice axis

    t = torch.stack((t,t,t)) # create fake color channel
    t = t.unsqueeze(0).float() # create batch dim

    magintude_y = random.randint(5, 25)
    magintude_x = random.randint(5, 25) # magnitude 5 is equal magintude 0.2 for fastai warp class

    warp_x = random.randint(-x//magintude_x, x//magintude_x) # no warping along the z axis (wraping only on 2D slices)
    warp_y = random.randint(-y//magintude_y, y//magintude_y)

    warp_x1 = round(float(x)/2 + warp_x)
    warp_x2 = x - warp_x1

    warp_y1 = round(float(y)/2 + warp_y)
    warp_y2 = y - warp_y1


    x = torch.cat(
        (torch.linspace(-1, 0, warp_x1),
        torch.linspace(0, 1, warp_x2+1)[1:]))
    y = torch.cat(
        (torch.linspace(-1, 0, warp_y1),
        torch.linspace(0, 1, warp_y2+1)[1:]))# without the +1)[1:] two 0 would be in the merged linspaces
    z = torch.linspace(-1, 1, z)

    meshx, meshy, meshz = torch.meshgrid((x, y, z)) #
    grid = torch.stack((meshy, meshx , meshz), 3) # create flow field. x and y need to be switched as otherwise the images are rotated.
    grid = grid.unsqueeze(0) # add batch dim
    out = F.grid_sample(t, grid, align_corners=True, mode = 'bilinear') # rescale the 5D tensor
    out = out[0,0,:,:,:].permute(2,0,1).contiguous() # remove fake color channels and batch dim, reorder the image (the Z axis has moved to the back...)
    return retain_type(out, typ = TensorDicom3D)

class RandomWarp3D(RandTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)

    def encodes(self, x:TensorDicom3D): return x.warp_3d()


# Cell

@patch
def add_gaussian_noise(t:TensorDicom3D, std):
    return t + (std**0.5)*torch.randn(t.shape)

class RandomNoise3D(RandTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        low_std = float(random.randint(1,20))/100
        high_std = float(random.randint(20,30))/100
        self.std = random.choice((low_std, low_std, low_std, low_std, high_std)) # lower noise is oversampled, as high noise migh be bad for the model

    def encodes(self, x:TensorDicom3D): return x.add_gaussian_noise(self.std)

# Cell

@patch
def rescale(t: TensorDicom3D, new_min = 0, new_max = 1):
    return (new_max - new_min)/(t.max()-t.min()) *(t - t.max()) + t.max()

@patch
def adjust_brightness(x:TensorDicom3D, beta):

    old_min = x.min()
    old_max = x.max()
    x = rescale(x, 0, 1)

    arr = x.numpy()
    arr = np.clip(arr + beta, arr.min(), arr.max())
    x = TensorDicom3D(arr)

    return rescale(x, old_min, old_max)


class RandomBrightness3D(RandTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.beta = float(random.randint(-300,400))/1000

    def encodes(self, x:TensorDicom3D):
        return x.adjust_brightness(self.beta)



# Cell

@patch
def adjust_contrast(x:TensorDicom3D, alpha):

    old_min = x.min()
    old_max = x.max()
    x = rescale(x, 0, 1)

    arr = x.numpy()
    arr = np.clip(arr * alpha, arr.min(), arr.max())
    x = TensorDicom3D(arr)

    return rescale(x, old_min, old_max)


class RandomContrast3D(RandTransform):
    def __init__(self, p=0.6):
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.alpha = float(random.randint(940, 1150))/1000

    def encodes(self, x:TensorDicom3D):
        return x.adjust_contrast(self.alpha)