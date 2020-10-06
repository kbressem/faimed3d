# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05-Custom-3D-CNN.ipynb (unless otherwise specified).

__all__ = ['Sequential_', 'resnet_3d', 'DoubleConv', 'Down', 'Up', 'OutConv', 'UNet3D', 'DiceLossBinary',
           'MCCLossBinary', 'mcc_binary']

# Cell
# export

import torchvision, torch
from torch import nn, Tensor
import torch.nn.functional as F

# Cell
class Sequential_(nn.Sequential):
    "Similar to nn.Sequential, but copies input to cuda"
    def forward(self, input):
        for module in self:
            input = module(input.cuda())
        return input

# Cell
def resnet_3d(n_input, n_classes):
    return Sequential_(
        # 1st Conv Block
        nn.Conv3d(n_input, 128, kernel_size = (3,7,7), stride = (2, 2, 1), padding = (1, 3, 3), bias = True),
        nn.BatchNorm3d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),
        nn.Dropout3d(),

        # 2nd Conv Block
        nn.Conv3d(128, 256, kernel_size = (3,4,4), stride = (2, 2, 2), padding = (1, 1, 1), bias = True),
        nn.BatchNorm3d(256, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),
        nn.Dropout3d(),

        # 3rd Conv Block
        nn.Conv3d(256, 384, kernel_size = (3,1,1), stride = (1, 1, 1), padding = (0, 0, 0), bias = True),
        nn.BatchNorm3d(384, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),
        nn.Dropout3d(),

        # 1st Res Block
        nn.Conv3d(384, 512, kernel_size = (3,3,3), stride = (1, 1, 1), padding = (1, 1, 1), bias = True),
        nn.BatchNorm3d(512, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),

        nn.Conv3d(512, 512, kernel_size = (3,3,3), stride = (1, 1, 1), padding = (1, 1, 1), bias = True),
        nn.BatchNorm3d(512, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),

        nn.Conv3d(512, 512, kernel_size = (3,3,3), stride = (1, 1, 1), padding = (1, 1, 1), bias = True),
        nn.BatchNorm3d(512, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),

        nn.Conv3d(512, 512, kernel_size = (3,3,3), stride = (1, 1, 1), padding = (1, 1, 1), bias = True),
        nn.BatchNorm3d(512, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
        nn.ReLU(),
        nn.Dropout3d(),

        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(512, n_classes),
        nn.Softmax(dim = 1))

# Cell

# copied from who????

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential_(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = Sequential_(
            nn.MaxPool3d(kernel_size = (2, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Sequential_(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        return self.conv(x)

# Cell
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
#        print('x:', x.shape)
        x1 = self.inc(x)
#        print('x1:', x1.shape)
        x2 = self.down1(x1)
#        print('x2:', x2.shape)
        x3 = self.down2(x2)
#        print('x3:', x3.shape)
        x4 = self.down3(x3)
#        print('x4:', x4.shape)
        x5 = self.down4(x4)
#        print('x5:', x5.shape)

        x = self.up1(x5, x4)
#        print('x:', x.shape)
        x = self.up2(x, x3)
#        print('x:', x.shape)
        x = self.up3(x, x2)
#        print('x:', x.shape)
        x = self.up4(x, x1)
#        print('x:', x.shape)
        logits = self.outc(x)
#        print(logits.shape)
        return logits

# Cell
class DiceLossBinary():

    """
    Simple DICE loss as described in:
        https://arxiv.org/pdf/1911.02855.pdf

    Computes the Sørensen–Dice loss. Larger is better.
    Note that PyTorch optimizers minimize a loss. So the loss is subtracted from 1.

    Args:
        targ: a tensor of shape [B, 1, D, H, W].
        pred: a tensor of shape [B, C, D, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability (acoid division by 0).
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """

    def __init__(self, method = 'miletari', alpha = 0.5, beta = 0.5, eps = 1e-7, smooth = 1.) -> None:
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.smooth = smooth

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        if input.min() < 0 or input.max() > 1:
            warn("Input is not in range between 0 and 1 but the loss will work better with input in that range. Consider rescaling your input. ")

        dims = (0,) + tuple(range(2, target.ndim))

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


class MCCLossBinary(DiceLossBinary):

    """
    Computes the MCC loss. Larger is better.
    For this loss to work best, the input should be in range 0-1, e.g. enforced through a sigmoid or softmax.
    Note that PyTorch optimizers minimize a loss. So the loss is subtracted from 1.
    Args:
        input: a tensor of shape [B, 1, D, H, W].
        target: a tensor of shape [B, C, D, H, W]. Corresponds to
            the raw output or logits of the model.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        if input.min() < 0 or input.max() > 1:
            warn("Input is not in range between 0 and 1 but the loss will work better with input in that range. Consider rescaling your input. ")

        dims = (0,) + tuple(range(2, target.ndim))

        tps = torch.sum(input * target, dims) + self.smooth  # geht nur wenn preds zwischen 0 und 1 sind, softmax am ende des unets sollte noch gemacht werden
        fps = torch.sum(input * (1 - target), dims)
        fns = torch.sum((1 - input) * target, dims)
        tns = torch.sum((1 - input) * (1-target), dims)

        numerator = (tps * tns - fps * fns) + self.smooth
        denominator =  ((tps + fps) * (tns + tns) * (fps + tns) * (tps + fns))**0.5 + self.smooth

        mcc_loss = numerator / (denominator + self.eps)

        return 1-mcc_loss

# Cell
from torch import tensor

def mcc_binary(input, target, thres = 0.5):
    dims = (0,) + tuple(range(2, target.ndim))
    input = torch.where(input > thres, tensor(1.).cuda(), tensor(0.).cuda())
    mcc = MCCLossBinary(smooth = 0.)(input, target)

    return torch.mean(1-mcc)