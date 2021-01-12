# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/08_parallel.unet.ipynb (unless otherwise specified).

__all__ = ['UnetBlock3D']

# Cell
# default_exp parallel
from .all import *
from fastai.vision.all import create_body, create_unet_model, hook_outputs, DynamicUnet
from fastai.vision.learner import _default_meta, _add_norm, model_meta
from fastai.vision.models.unet import  _get_sz_change_idxs



# Cell
class UnetBlock3D(nn.Module):
    "A quasi-UNet block, using `ConvTranspose3d` for upsampling`."
    @delegates(ConvLayer.__init__)
    def __init__(self, up_in_c, x_in_c, final_div=True, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        self.up = ConvTranspose3D(up_in_c, up_in_c//2, blur=blur, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.bn = BatchNorm(x_in_c, ndim=3)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = ConvLayer(ni, nf, ndim=3, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, ndim=3, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in, lwr_features):
        up_out = self.up(up_in)
        ssh = lwr_features.shape[-3:]
        if ssh != up_out.shape[-3:]:
            up_out = F.interpolate(up_out, lwr_features.shape[-3:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(lwr_features)], dim=1))
        return self.conv2(self.conv1(cat_x))
