# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_layers.ipynb (unless otherwise specified).

__all__ = ['in_channels', 'num_features_model', 'model_sizes_4d', 'dummy_eval_4d', 'RecurrentSequential', 'Arch4D',
           'Concat', 'AdaptiveConcatPool3d', 'Sequential4D', 'create_head', 'create_cnn_model_3d']

# Cell
# default_exp layers
from .basics import *
from .augment import *
from .preprocess import *
from .models.resnet import resnet50_3d
from .data import *
from fastai.vision.learner import _default_meta, _add_norm, model_meta, create_body
from fastai.layers import ResBlock

# Cell
from fastai.basics import *
from fastai.callback.all import *

# Cell
def in_channels(m):
    """
    Return the shape of the first weight layer in `m`.
    same as fastai.vision.learner.in_channels but allows l.weight.ndim of 4 and 5
    """
    for l in flatten_model(m):
        if getattr(l, 'weight', None) is not None and l.weight.ndim in [4,5]:
            return l.weight.shape[1]
    raise Exception('No weight layer')

# Cell
def num_features_model(m):
    """
    Return the number of output features for `m`.
    same as fastai.vision.learner.num_features_model passes model_size a len 3 tuple of sz

    """
    sz,ch_in = 32,in_channels(m)
    while True:
        #Trying for a few sizes in case the model requires a big input size.
        return model_sizes_4d(m, (sz,sz,sz))[-1][1]
        try:
            return model_sizes_4d(m, (sz,sz,sz))[-1][1]
        except Exception as e:
            sz *= 2
            print(sz)
            if sz > 2048: raise e

# Cell
def model_sizes_4d(m, size=(8,64,64), n_inp=1):
    "Pass a dummy input through the model `m` to get the various sizes of activations. same as fastai func"
    with hook_outputs(m) as hooks:
        _ = dummy_eval_4d(m, size=size, n_inp=n_inp)
        return [o.stored[0].shape for o in hooks]

# Cell
def dummy_eval_4d(m, size=(8,64,64), n_inp=1):
    "Evaluate `m` on a dummy input of a certain `size`. Same as fastai func"
    ch_in = in_channels(m)
    x = one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)
    with torch.no_grad(): return m.eval()((x, )*n_inp)

# Cell
class RecurrentSequential(nn.Sequential):
    "Makes some parts of the network recurrent"
    def __init__(self, *args):
        "reduces some of the hierachical structure of the network"
        super().__init__(*[submodule for module in args for submodule in module])

    def forward(self, inputs):
        for module in self:
            out = None
            for i, inp in enumerate(inputs):
                if isinstance(out, Tensor) and out.shape == inp.shape: inp += out
                out = module(inp)
                inputs[i]=out
        return inputs

# Cell
class Arch4D(SequentialEx):
    'repeatedly applies the same network to different inputs'
    def __init__(self, arch, n_inp):
        layers = [RecurrentSequential(l) for l in arch]
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs)->list:
        inputs = list(inputs)
        for l in self.layers:
            inputs = l(inputs)
        return inputs

# Cell
class Concat(Module):
    def __init__(self, ni, ndim, dim = 1):
        store_attr()
        self.bn = BatchNorm(ni, ndim)
        self.act = nn.ReLU()

    def forward(self, inputs:(list, tuple))->Tensor:
        inputs = torch.cat(inputs, self.dim)
        return self.act(self.bn(inputs))

# Cell
class AdaptiveConcatPool3d(Module):
    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool3d(self.size)
        self.mp = nn.AdaptiveMaxPool3d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# Cell
class Sequential4D(nn.Sequential):
    def forward(self, *input):
        input = list(input)
        for module in self:
            input = module(input)
        return input

# Cell
def create_head(nf, n_out, n_inp=None, lin_ftrs=None, ps=0.5, concat_pool=True, bn_final=False, lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool3d() if concat_pool else nn.AdaptiveAvgPool3d(1)
    if n_inp is not None:
        pool = nn.Sequential(Concat(n_inp*nf//2, 3),
                             # nf is not the true number of features but number of features * 2
                             # therefore we need to divide by 2 in Concat
                             pool,
                             ConvLayer(n_inp*nf, nf, ndim = 3, ks = 1, stride = 1))
    layers = [pool, Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)

# Cell
def create_cnn_model_3d(arch, n_out, n_inp, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom convnet architecture using `arch`, `n_in` and `n_out`. Identical to fastai func"
    body = create_body(arch, n_in, pretrained, cut)
    body = Arch4D(body, n_inp)
    if custom_head is None:
        nf = num_features_model(body) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, n_inp, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = Sequential4D(body, head)
    if init is not None: apply_init(model[1], init)
    return model