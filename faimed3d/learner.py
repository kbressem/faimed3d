# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_learner.ipynb (unless otherwise specified).

__all__ = ['cnn_learner_3d', 'create_unet_model_3d', 'unet_learner_3d']

# Cell
# default_exp learner
from fastai.basics import *
from fastai.callback.all import *

# Cell
from .basics import *
from .augment import *
from .preprocess import *
from .layers import *
from .data import *
from .models.unet import DynamicUnet3D
from .models.losses import DiceLoss
from fastai.vision.learner import _default_meta, _add_norm, model_meta, create_body, create_unet_model

# Cell
@delegates(Learner.__init__)
def cnn_learner_3d(dls, arch, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    """
    Build a convnet style learner from `dls` and `arch`
    Same as fastai func but adds the `AddColorChannel` callback.
    """
    if config is None: config = {}
    meta = model_meta.get(arch, _default_meta)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    # if normalize: _add_norm(dls, meta, pretrained) # no effect as no TenosrImage is passed in 3d
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_cnn_model(arch, n_out, ifnone(cut, meta['cut']), pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=ifnone(splitter, meta['split']), **kwargs)
    if pretrained: learn.freeze()
    return learn

# Cell
@delegates(DynamicUnet3D.__init__)
def create_unet_model_3d(arch, n_out, img_size, pretrained=True, cut=None, n_in=3, **kwargs):
    "Create custom unet architecture"
    meta = model_meta.get(arch, _default_meta)
    body = create_body(arch, n_in, pretrained, ifnone(cut, meta['cut']))
    model = DynamicUnet3D(body, n_out, img_size, **kwargs)
    return model

# Cell
@delegates(create_unet_model)
def unet_learner_3d(dls, arch, normalize=True, n_out=None, pretrained=True, config=None,
                 # learner args
                 loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
                 model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
                 # other model args
                 norm_type=NormType.Batch, **kwargs):
    "Build a unet learner from `dls` and `arch`"

    if config:
        warnings.warn('config param is deprecated. Pass your args directly to unet_learner.')
        kwargs = {**config, **kwargs}

    meta = model_meta.get(arch, _default_meta)
    # if normalize: _add_norm(dls, meta, pretrained) # no effect as no TenosrImage is passed in 3d

    n_out = ifnone(n_out, get_c(dls))
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    img_size = dls.one_batch()[0].shape[-3:]
    assert img_size, "image size could not be inferred from data"
    model = create_unet_model_3d(arch, n_out, img_size, pretrained=pretrained, norm_type=norm_type, **kwargs)

    if loss_func is None: loss_func = DiceLoss(smooth=0.)
    splitter=ifnone(splitter, meta['split'])
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
                   moms=moms)
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn)
    if kwargs: store_attr(self=learn, **kwargs)
    return learn