# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06f_models.losses.ipynb (unless otherwise specified).

__all__ = ['BaseLoss', 'DiceLoss', 'MCCLoss']

# Cell
# export
from fastai.basics import *
from ..basics import *
import torchvision, torch
from warnings import warn

# Cell
class BaseLoss():

    """
    Base class for loss functions

    Args:
        targ:    A tensor of shape [B, C, D, H, W].
        pred:    A tensor of shape [B, C, D, H, W]. Corresponds to
                 the raw output or logits of the model.
        weights: A list ot tuple, giving weights for each class or None
        avg:     'macro' computes loss for each B x C and averages the losses
                 'micro' computes loss for each B and acverages the losses
    Returns:
        loss:    computed loss (scalar)
    """

    def __init__(self, weights = None, avg = 'macro') -> None:
        store_attr()

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        if input.min() < 0 or input.max() > 1:
            input = F.softmax(input, dim = 1)

       #     warnings.warn('Range of input values is between {} and {} but should be between 0 and 1. '
       #                   'Softmax function will be applied to the channel dim'.format(input.min().detach(),
       #                                                                                input.max().detach()))

        if target.size(1) != input.size(1):
            target = self.one_hot(target,input.size(1))

        dims = (2,3,4) if self.avg == 'macro' else (1,2,3,4)

        loss_per_item_and_channel =self.compute_loss(input, target, dims)

        if self.weights != None:
            self.weights = tensor(self.weights).to(input.device)
            if self.weights.size(0) != loss_per_item_and_channel.size(1):
                raise ValueError('Number of weights does not match number of classes.')
            loss_per_item_and_channel = loss_per_item_and_channel * self.weights
        loss = torch.mean(loss_per_item_and_channel)
        return loss if loss > 0 else 1-loss # loss can sometimes become negative

    def compute_loss(self, input, target, dims) -> Tensor:
        pass

    def one_hot(self, mask, n_classes) -> Tensor:
        return torch.cat([(mask == i).float() for i in range(0, n_classes)], 1)

# Cell
class DiceLoss(BaseLoss):

    """
    Simple DICE loss as described in:
        https://arxiv.org/pdf/1911.02855.pdf

    Computes the Sørensen–Dice loss. Larger is better.
    Note that PyTorch optimizers minimize a loss. So the loss is subtracted from 1.

    Args:
        inherited from `BaseLoss`
        targ:    A tensor of shape [B, C, D, H, W].
        pred:    A tensor of shape [B, C, D, H, W]. Corresponds to
                 the raw output or logits of the model.
        weights: A list ot tuple, giving weights for each class or None
        avg:     'macro' computes loss for each B x C and averages the losses
                 'micro' computes loss for each B and acverages the losses

        Unique for `DiceLoss`
        method:  The method, how the DICE score should be calcualted.
                    "simple"   = standard DICE loss
                    "miletari" = squared denominator for faster convergence
                    "tversky"  = variant of the DICE loss which allows to weight FP vs FN.
        alpha, beta: weights for FP and FN for "tversky" loss, if both values are 0.5 the
                 "tversky" loss corresponds to the "simple" DICE loss
        smooth:  Added smoothing factor.
        eps: added to the denominator for numerical stability (acoid division by 0).
    Returns:
        dice_loss: the Sørensen–Dice loss
    """

    def __init__(self, method = 'miletari', alpha = 0.5, beta = 0.5, eps = 1e-7, smooth = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        store_attr()

    def compute_loss(self, input, target, dims):
        if self.method == 'simple':
            numerator  = torch.sum(input * target, dims) + self.smooth
            denominator  = torch.sum(input + target, dims) + self.smooth
            dice_loss = (2. * numerator / (denominator + self.eps))

        elif self.method == 'miletari':
            numerator  = torch.sum(input * target, dims) + self.smooth
            denominator  = torch.sum(input**2 + target**2, dims) + self.smooth
            dice_loss = (2. * numerator / (denominator + self.eps))

        elif self.method == 'tversky':
            numerator  = torch.sum(input * target, dims) + self.smooth
            fps = torch.sum(input * (1 - target), dims)
            fns = torch.sum((1 - input) * target, dims)

            denominator  = numerator + self.alpha*fps + self.beta*fns + self.smooth
            dice_loss = (2. * numerator / (denominator + self.eps))

        else:
            raise NotImplementedError('The specified type of DICE loss is not implemented')

        return 1-dice_loss



# Cell
class MCCLoss(BaseLoss):

    """
    Computes the MCC loss.

    For this loss to work best, the input should be in range 0-1, e.g. enforced through a sigmoid or softmax.
    Note that PyTorch optimizers minimize a loss. So the loss is subtracted from 1.
    While the MCC score can become negative, the MCC loss should not go below 0


    Args:
        inherited from `BaseLoss`
        targ:    A tensor of shape [B, C, D, H, W].
        pred:    A tensor of shape [B, C, D, H, W]. Corresponds to
                 the raw output or logits of the model.
        weights: A list ot tuple, giving weights for each class or None
        avg:     'macro' computes loss for each B x C and averages the losses
                 'micro' computes loss for each B and acverages the losses

        Unique for `MCCLoss`
        smooth:  Smoothing factor, default is 1.
        eps:     Added for numerical stability.

    Returns:
        mmc_loss: loss based on Matthews correlation coefficient
    """
    def __init__(self, eps=1e-7, smooth=1.,*args, **kwargs):
        store_attr()
        super().__init__(*args, **kwargs)

    def compute_loss(self, input: Tensor, target: Tensor, dims):

        tps = torch.sum(input * target, dims)
        fps = torch.sum(input * (1 - target), dims)
        fns = torch.sum((1 - input) * target, dims)
        tns = torch.sum((1 - input) * (1-target), dims)

        numerator = (tps * tns - fps * fns) + self.smooth
        denominator =  ((tps + fps) * (tps + fns) * (fps + tns) * (tns + fns) + self.eps)**0.5 + self.smooth

        mcc_loss = numerator / (denominator)

        return 1-mcc_loss