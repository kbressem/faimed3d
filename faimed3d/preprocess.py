# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_preprocessing.ipynb (unless otherwise specified).

__all__ = ['MeanScale', 'ImageCorrectionWrapper', 'get_percentile', 'get_landmarks', 'find_standard_scale',
           'PiecewiseHistScaling', 'standard_scale_from_filelist']

# Cell
# default_exp preprocess
import SimpleITK as sitk
from fastai.basics import *
from fastai.vision.augment import *
from .basics import *
from fastai.torch_core import interp_1d
from tqdm import tqdm

# Cell
@patch
def size_correction(im:(TensorDicom3D, TensorMask3D), new_spacing=1):
    x_sz, y_sz, z_sz  = im.get_spacing()
    m = im.metadata
    rescale_factor = new_spacing/x_sz
    new_sz = (im.size(-3),
              int(im.size(-2)*rescale_factor),
              int(im.size(-1)*rescale_factor))
    mode = 'trilinear' if isinstance(im, TensorDicom3D) else 'nearest'
    while im.ndim < 5: im = im.unsqueeze(0)
    im = F.interpolate(im, size = new_sz, mode = mode, align_corners=True).squeeze() #changes memory address, restore_metadata won't work anymore
    im.metadata = m
    im.set_spacing((new_spacing, new_spacing, z_sz))
    return im

# Cell
@patch
def rescale_pixeldata(t:TensorDicom3D):
    m = t.metadata
    if '0028|1053' in m: # if one tag is present, the other should also
        t = t * m['0028|1053'] + m['0028|1052']
        m.pop('0028|1052')
        m.pop('0028|1053')
        t.restore_metadata()
    return t

# Cell
@patch
def mean_scale(t:TensorDicom3D):
    "Scales pixels by subtracting the mean and dividing by std. 0 pixels are ignored"
    t = t - t.min() # set mit to 0
    mask = t.ne(0.)
    mean, sd = t[mask].mean(), t[mask].std()
    t = (t - mean) / sd
    t.restore_metadata()
    return t

class MeanScale(RandTransform):
    split_idx,order = None, 10
    def __init__(self, p=1., **kwargs):
        super().__init__(p, **kwargs)
        store_attr()
    def encodes(self, x:TensorDicom3D):
        return x.mean_scale()
    def encodes(self, x:TensorMask3D): return x

# Cell
@patch
def median_scale(t:TensorDicom3D):
    "Scales pixels by subtracting the median and dividing by the IQR. 0 pixels are ignored"
    t = t - t.min() # set mit to 0
    mask = t.ne(0.)
    if mask.view(-1).shape[0] >=2 **16:
        # resize to large tensor for quantile
        # qunatile takes up to size 2**24, but than takes ~1sec
        mask = F.interpolate(t[mask].view(-1).unsqueeze(0).unsqueeze(0), 2**16)
    median, iqr = mask.median(), mask.quantile(0.75)-mask.quantile(0.25)
    t = (t-median)/iqr
    t.restore_metadata()
    return t

# Cell
@patch
def max_scale(t:TensorDicom3D):
    t = (t - t.min()) / (t.max() - t.min())
    t.restore_metadata()
    return t

# Cell
@patch
def freqhist_bins(t:(TensorDicom3D,Tensor), n_bins=100):
    '''
    A function to split the range of pixel values into groups, such that each group has around the same number of pixels.
    taken from https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py#L78
    '''
    imsd = t.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()

@patch
def hist_scaled(t:(TensorDicom3D,Tensor), brks=None):
    '''
    Scales a tensor using `freqhist_bins` to values between 0 and 1
    taken from https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py#L78
    '''
    if t.device.type=='cuda': return t.hist_scaled_pt(brks)
    if brks is None: brks = t.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = t.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    x = tensor(x).reshape(t.shape).clamp(0.,1.)
    return x # will loose meta data in process

@patch
def hist_scaled_pt(t:(TensorDicom3D,Tensor), brks=None):
    "same as fastai fucntion for PILDicom"
    # Pytorch-only version - switch to this if/when interp_1d can be optimized
    if brks is None: brks = t.freqhist_bins()
    brks = brks.to(t.device)
    ys = torch.linspace(0., 1., len(brks)).to(t.device)
    return t.flatten().interp_1d(brks, ys).reshape(t.shape).clamp(0.,1.)

# Cell
class ImageCorrectionWrapper(object):
    def __init__(self,
                 n4_max_num_it = 3,
                 hist_radius = [5,5,5], # radius in format [H x W x D]. Computation time scale ^3 with radius.
                 hist_alpha  = 0.3,
                 hist_beta = 0.3,
                 do_n4 = True,
                 do_hist = True,
                 verbose = True):
        store_attr()
        if do_n4:
            self.corrector = sitk.N4BiasFieldCorrectionImageFilter()

    def __call__(self, orig_file, fn_out=None):
        if isinstance(orig_file, str):
            if fn_out is None:
                fn_out = self.strip_suffix(fn_in)+'corrected.nii.gz'
            self.convert_string(orig_file, fn_out)
        if isinstance(orig_file, TensorDicom3D):
            if fn_out is None:
                try: fn_out = self.strip_suffix(orig_file.fn)+'corrected.nii.gz'
                except: raise ValueError('Please set a name for the output')
            self._convert(orig_file.as_sitk(), fn_out)
        if self.verbose:
            print('Coreccted and wrote file to {}'.format(fn_out))

    def convert_string(self, fn_in, fn_out):
        im = self.read_image(fn_in)
        self._convert(im, fn_out)

    def _convert(self, im, fn_out):
        if self.do_n4:
            im = self.n4_bias_correction(im)
        if self.do_hist:
            im = self.hist_equal(im)
        sitk.WriteImage(im, fn_out)


    def n4_bias_correction(self, im):
        self.corrector.SetMaximumNumberOfIterations([self.n4_max_num_it]*3)
        return self.corrector.Execute(im)

    def hist_equal(self, im):
        return sitk.AdaptiveHistogramEqualization(sitk.Cast(im, sitk.sitkInt16),
                                                  radius=self.hist_radius*3,
                                                  alpha=self.hist_alpha,
                                                  beta=self.hist_beta)

    def read_image(self, fn):
        "copy of TensorDicom3D.load"
        if isinstance(fn, str): fn = Path(fn)
        if fn.is_dir():
            SeriesReader = sitk.ImageSeriesReader()
            dicom_names = SeriesReader.GetGDCMSeriesFileNames(str(fn))
            SeriesReader.SetFileNames(dicom_names)
            im = SeriesReader.Execute()
            return sitk.Cast(im, sitk.sitkFloat32)
        elif fn.is_file():
            return sitk.ReadImage(str(fn), outputPixelType=sitk.sitkFloat32)
        else:
            raise TypeError('the path "{}" is neither a valid directory nor a file'.format(str(fn)))


    def strip_suffix(self, fn):
        fn = Path(fn)
        extensions = "".join(fn.suffixes)
        new_fn = str(fn).replace(extensions, '')
        return new_fn+'/' if fn.is_dir() else new_fn

# Cell
def get_percentile(t: Tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (float).

    This function is twice as fast as torch.quantile and has no size limitations
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k)[0].item()

    return result

# Cell
def get_landmarks(t: Tensor, percentiles: Tensor)->Tensor:
    """
    Returns the input's landmarks.

    :param t (torch.Tensor): Input tensor.
    :param percentiles (torch.Tensor): Peraentiles to calculate landmarks for.
    :return: Resulting landmarks (torch.tensor).
    """
    return tensor([get_percentile(t, perc.item()) for perc in percentiles])

# Cell
def find_standard_scale(inputs, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images
    Args:
        inputs (list or L): set of TensorDicom3D objects which are to be normalized
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    percs = torch.cat([torch.tensor([i_min]),
                       torch.arange(l_percentile, u_percentile+1, step),
                       torch.tensor([i_max])], dim=0)
    standard_scale = torch.zeros(len(percs))

    for input_image in inputs:
        mask_data = input_image > input_image.mean()
        masked = input_image[mask_data > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = get_percentile(masked, i_min)
        max_p = get_percentile(masked, i_max)
        new_landmarks = landmarks.interp_1d(torch.FloatTensor([i_s_min, i_s_max]),
                                            torch.FloatTensor([min_p, max_p]))
        standard_scale += new_landmarks
    standard_scale = standard_scale / len(inputs)
    return standard_scale, percs

# Cell
@patch
def piecewise_hist(image:Tensor, landmark_percs, standard_scale)->Tensor:
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks
    Args:
        input_image (TensorDicom3D): image on which to find landmarks
        landmark_percs (torch.tensor): corresponding landmark points of standard scale
        standard_scale (torch.tensor): landmarks on the standard scale
    Returns:
        normalized (TensorDicom3D): normalized image
    """
    mask_data = image > image.mean()
    masked = image[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    if landmarks.device != image.device: landmarks = landmarks.to(image.device)
    if standard_scale.device != image.device: standard_scale = standard_scale.to(image.device)
    return image.flatten().interp_1d(landmarks, standard_scale).reshape(image.shape)

class PiecewiseHistScaling(RandTransform):
    split_idx,order = None, 10
    def __init__(self, landmark_percs=None, standard_scale=None, p=1., **kwargs):
        super().__init__(p, **kwargs)
        if landmark_percs is None or standard_scale is None:
            raise ValueError('Landmark parcs and standard scale nned to be provided.'
                             'You can run `standard_scale_from_filelist` or `standard_scale_from_dls` '
                             'To get an estiamtion of the values. Alternatively you can use the '
                             '`PiecewiseHistNormalizationCallback` which will automatically calulated '
                             'the needed values before the first epoch.')
        store_attr()

    def encodes(self, x:TensorDicom3D):
        x = x.piecewise_hist(self.landmark_percs, self.standard_scale)
        return (x - x.mean()) / x.std()

    def encodes(self, x:TensorMask3D): return x


# Cell
def standard_scale_from_filelist(fns:(list, pd.Series)):
    scales = []
    for fn in tqdm(fns):
        x = TensorDicom3D.create(fn)
        scale, percs = find_standard_scale(x)
        scales.append(scale)
    return torch.stack(scales).mean(0), percs

# Cell
@patch
def standard_scale_from_dls(dls:DataLoaders):
    "calculates standard scale from images in a dataloader"
    scales = []
    for i, pair in tqdm(enumerate(dls.train_ds)):
        x, _ = pair
        scale, percs = find_standard_scale(x)
        scales.append(scale)
    return torch.stack(scales).mean(0), percs