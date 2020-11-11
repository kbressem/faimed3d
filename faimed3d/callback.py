# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_callback.ipynb (unless otherwise specified).

__all__ = ['StackVolumes', 'SplitVolumes', 'SubsampleShuffle', 'MixSubvol', 'MixUp3D', 'ReloadBestFit']

# Cell
# default_exp callback

from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch.nn.functional as F

# Cell

from .basics import *
from .augment import *
from .data import *

# Cell
class StackVolumes(Callback):
    """
    Takes multiple 3D volumes and stacks them in the color dim.
    This is usefull when using mutli-sequence medical data.

    Can also merge multiple segmentation masks, through pooling (max, min or mean) alongside the color dim,
    then convertes the mask to one-hot encoded type. However, this can lead to 'ugly' masks with punch-out like
    appearences

    Example:
        Having the Tensors of size (10, 1, 5, 25, 25) would lead to a single Tensor of
        size (10, 3, 5, 25, 25).
    """
    def __init__(self, n_classes, pool_type='max', stack_yb=False):
        store_attr()

    def before_batch(self):
        self.learn.xb = (torch.cat(self.learn.xb, dim=1), )

        if self.stack_yb:
            if self.pool_type=='max':
                yb = torch.cat(self.learn.yb, dim = 1).max(dim=1)[0] # get max at color channel
            elif self.pool_type=='mean':
                yb = torch.cat(self.learn.yb, dim = 1).mean(dim=1)[0].round() # get mean at color channel
            elif self.pool_type=='min':
                yb = torch.cat(self.learn.yb, dim = 1).min(dim=1)[0] # get min at color channel

            else: raise NotImplementedError('Pooling type {} for mask not implemented'.format(self.pool_type))
            yb = self.to_one_hot(yb, self.n_classes)
            self.learn.yb = (yb, )

    def make_binary(self, target, val):
        return torch.where(target == val, tensor(1.).to(target.device), tensor(0.).to(target.device))

    def to_one_hot(self, target, n_classes):
        target = target.squeeze(1).long() # remove the solitary color channel (if there is one) and set type to int64
        one_hot = [self.make_binary(target, val=i) for i in range(0, n_classes)]
        return torch.stack(one_hot, 1)

# Cell
class SplitVolumes(Callback):
    """
        Separates a 3D tensor into smaller equal sized subvolumes.

         o---o---o       o---o---o
         | A | A |       | B | B |        o---o  o---o  o---o  o---o  o---o  o---o  o---o  o---o
         o---o---o   +   o---o---o  ==>   | A | +| A | +| B | +| B | +| A | +| A | +| B | +| B |
         | A | A |       | B | B |        o---o  o---o  o---o  o---o  o---o  o---o  o---o  o---o
         o---o---o       o---o---o


        Args:
            n_subvol = number of subvolumes
            split_along_depth = whether volumes should also be split along the D dimension fpr a [B, C, D, H, W] tensor
    """
    run_after = StackVolumes
    def __init__(self, n_subvol = 2**3, split_along_depth = True):
        store_attr()

    def before_batch(self):
        xb = self.learn.xb
        if len(xb) > 1: raise ValueError('Got multiple items in x batch. You need to concatenate the batch first.')
        self.learn.xb = self.split_volume(xb)
        self.learn.yb = self.split_volume(self.learn.yb)

    def after_pred(self):
        self.learn.xb = self.patch_volume(self.learn.xb)
        self.learn.pred = detuplify(self.patch_volume(self.learn.pred))
        self.learn.yb = self.patch_volume(self.learn.yb)

    def split_volume(self, xb:(Tensor, TensorDicom3D, TensorMask3D)):
        "splits a large tensor into multiple smaller tensors"

        xb = detuplify(xb) # xb is always a tuple
        # calculate number of splits per dimension
        self.n = self.n_subvol**(1/3) if self.split_along_depth else self.n_subvol**0.5
        self.n = int(self.n)

        # check if shape of dims is divisible by n, if not resize the Tensor acordingly
        shape = [s if s % self.n == 0 else s - s % self.n for s in xb.shape[-3:]]
        if not self.split_along_depth: shape[0]=xb.shape[0]
        xb = F.interpolate(xb, size = shape, mode = "trilinear", align_corners=True)

        # split each dim into smaller dimensions
        d, h, w = shape
        if self.split_along_depth: xb = xb.reshape(xb.size(0), xb.size(1), self.n, int(d/self.n), self.n, int(h/self.n), self.n, int(w/self.n))
        else: xb = xb.reshape(xb.size(0), xb.size(1),1, d, self.n, int(h/self.n), self.n, int(w/self.n))

        # swap the dimensions an flatten Batchdim and the newly created dims
        # return a tuple as xb is always a tuple
        return (xb.permute(1, 3, 5, 7, 0, 2, 4, 6).flatten(-4).permute(4, 0, 1, 2, 3), )

    def patch_volume(self, p:(Tensor, TensorDicom3D, TensorMask3D)):
        "patches a prior split volume back together"
        p = detuplify(p)

        old_shape = p.shape[0]//self.n_subvol, p.shape[1], *[s * self.n for s in p.shape[2:]]
        if not self.split_along_depth: old_shape[2]=p.shape[2]
        p = p.reshape(old_shape[0], self.n, self.n, self.n, *p.shape[1:])
        return (p.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(old_shape), )

# Cell
class SubsampleShuffle(SplitVolumes):
    """
        After splitting rhe volume into multiple subvolumes, draws a radnom amount of subvolumes for training.
        Would allow to train on an effective batch size < 1.

        o---o---o        o---o---o
        | A | A |        | B | B |        o---o  o---o  o---o  o---o  o---o  o---o
        o---o---o    +   o---o---o  ==>   | B | +| A | +| A | +| A | +| B | +| A |
        | A | A |        | B | B |        o---o  o---o  o---o  o---o  o---o  o---o
        o---o---o        o---o---o

        Args:
            p: percentage of subvolumes to train on
    """
    run_after = [StackVolumes]

    def __init__(self, p = 0.5, n_subvol=2**3, split_along_depth = True):
        store_attr()

    def before_batch(self):

        xb = self.learn.xb
        if len(xb) > 1: raise ValueError('Got multiple items in x batch. You need to concatenate the batch first.')
        self.learn.xb = self.split_volume(xb)
        self.learn.yb = self.split_volume(self.learn.yb)

        if self.training:
            xb = detuplify(self.learn.xb)
            yb = detuplify(self.learn.yb)
            draw = tuple(random.sample(range(0, xb.size(0)), int(xb.size(0)*self.p)))
            self.learn.xb = (xb[draw, :], )
            self.learn.yb = (yb[draw, :], )

    def after_pred(self):
        if not self.training:
            self.learn.xb = self.patch_volume(self.learn.xb)
            self.learn.pred = detuplify(self.patch_volume(self.learn.pred))
            self.learn.yb = self.patch_volume(self.learn.yb)

# Cell
class MixSubvol(SplitVolumes):
    """
        After splitting rhe volume into multiple subvolumes, shuffels the subvolumes and sticks the images back together.

        o---o---o        o---o---o        o---o---o        o---o---o
        | A | A |        | B | B |        | B | B |        | A | B |
        o---o---o    +   o---o---o  ==>   o---o---o    +   o---o---o
        | A | A |        | B | B |        | A | A |        | B | A |
        o---o---o        o---o---o        o---o---o        o---o---o


        Args:
            p: probability that the callback will be applied
            n_subvol: number of subvolumina to create
            split_along_depth: whether the depth dimension should be included

    """
    run_after = [StackVolumes]

    def __init__(self, p = 0.25, n_subvol=2**3, split_along_depth = True):
        store_attr()

    def before_batch(self):
        if self.training and random.random() < self.p:
            xb = self.learn.xb
            if len(xb) > 1: raise ValueError('Got multiple items in x batch. You need to concatenate the batch first.')
            xb = detuplify(self.split_volume(xb))
            yb = detuplify(self.split_volume(self.learn.yb))
            shuffle = tuple(random.sample(range(0, xb.size(0)), xb.size(0)))
            self.learn.xb = self.patch_volume((xb[shuffle, :], ))
            self.learn.yb = self.patch_volume((yb[shuffle, :], ))

    def after_pred(self):
        pass


# Cell
class MixUp3D(Callback):
    """
    Implementation of MixUp for 3D images.
    Note that the loss function does not need to be adapted like in fastais MixUp, as MCC and DICE loss accept float values.
    """

    run_after = [Normalize, StackVolumes]
    def __init__(self, p = 0.5):
        store_attr()

    def before_batch(self):
        if self.training and random.random() < self.p:
            if len(self.learn.xb) > 1: raise ValueError('Got multiple items in x batch. You need to concatenate the batch first.')
            shuffled_idx = list(range(0, detuplify(self.learn.xb).size(0)))
            random.shuffle(shuffled_idx)
            xj = detuplify(self.learn.xb)[shuffled_idx, :]
            yj = detuplify(self.learn.yb)[shuffled_idx, :]
            w = random.random()
            self.learn.xb = (detuplify(self.learn.xb)*w + xj*(1-w), )
            self.learn.yb = (detuplify(self.learn.yb)*w + yj*(1-w), )


# Cell
class ReloadBestFit(TrackerCallback):
    "A `TrackerCallback` that reloads the previous best model if not improvement happend for n epochs"
    def __init__(self, fname,  monitor='valid_loss', comp=None, min_delta=0., patience=1):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.patience = patience
        self.fname = fname

    def before_fit(self): self.wait = 0; super().before_fit()
    def after_epoch(self):
        "Compare the value monitored to its best score and maybe stop training."
        super().after_epoch()
        if self.new_best: self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'No improvement since epoch {self.epoch-self.wait}: reloading previous best model.')
                self.learn = self.learn.load(self.fname)
                self.wait=0