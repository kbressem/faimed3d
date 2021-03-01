# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06c_model.efficientnet.ipynb (unless otherwise specified).

__all__ = ['ConvLayerDynamicPadding', 'DropConnect', 'MBConvBlock', 'EfficientNet', 'efficientnet_b0',
           'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
           'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'efficientnet_l2']

# Cell
# export
from fastai.basics import *
from torch.hub import load_state_dict_from_url

# Cell
class ConvLayerDynamicPadding(nn.Sequential):
    "Same as fastai ConvLayer, but more accurately padds input according to `ks` and `stride`"
    @delegates(nn.Conv3d)
    def __init__(self,
                 ni, #number of input channels
                 nf, # number of output channels
                 ks=3, # kernel size (tuple or int)
                 stride=1, # kernel stride (tuple or int)
                 bias=None, # bias of convolution
                 ndim=3, # dimension of convolution (1,2,3)
                 norm_type=NormType.Batch, # type of batch nornalization
                 bn_1st=True, # batch norm before ReLU
                 act_cls=defaults.activation, # activation function
                 transpose=False, # if transpose convolution should be constructed
                 init='auto', # type of initialization
                 xtra=None, # extra layers
                 bias_std=0.01,
                 **kwargs # further arguments for ConvLayer
                ):

        # asymmetric padding
        if isinstance(ks, int): ks = (ks, )*ndim
        padding = [pad for _ks in ks for pad in self.calculate_padding(_ks)]

        # init ConvLayer but set padding to 0
        conv_layer = ConvLayer(ni, nf, ks, stride, 0, bias, ndim, norm_type, bn_1st, act_cls, transpose, init, xtra, bias_std, **kwargs)

        # set padding layer to first place, then all other layers
        # padding needs to be reverted, as the function expects format (W, W, H, H, D, D)
        super().__init__(nn.ConstantPad3d(padding[::-1], 0.),
                         *[l for l in conv_layer])

    def calculate_padding(self, ks):
        if ks % 2 == 0: return ks // 2, (ks-1) //2
        else: return ks //2, ks //2

# Cell
class DropConnect(nn.Module):
    "Drops connections with probability p"
    def __init__(self, p):
        assert 0 <= p <= 1, 'p must be in range of [0,1]'
        self.keep_prob = 1 - p
        super().__init__()

    def forward(self, x):
        if not self.training: return x
        batch_size = x.size(0)

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = self.keep_prob + torch.rand([batch_size, 1, 1, 1, 1], dtype=x.dtype, device=x.device)
        return x / self.keep_prob * random_tensor.floor_() # convert random tensor to binary tensor

# Cell
class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self,
                 n_inp, # number of input channels
                 n_out, # number of output channels
                 kernel_size, # size of convolution kernel
                 stride, # stride of kernel
                 se_ratio, # squeeze-expand ratio
                 id_skip, # if skip connection shouldbe used
                 expand_ratio, # expansion ratio for inverted bottleneck
                 drop_connect_rate = 0.2, # percentage of dropped connections
                 act_cls=nn.SiLU, # type of activation function
                 norm_type=NormType.Batch, # type of batch normalization
                 **kwargs # further arguments passed to `ConvLayerDynamicPadding`
                ):
        super().__init__()
        store_attr()

        # expansion phase (inverted bottleneck)
        n_intermed = n_inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self.expand_conv = ConvLayerDynamicPadding(ni=n_inp, nf=n_intermed,
                                                       ks = 1,norm_type=norm_type,
                                                       act_cls=act_cls, **kwargs)

        # depthwise convolution phase, groups makes it depthwise
        self.depthwise_conv = ConvLayerDynamicPadding(ni=n_intermed, nf=n_intermed,
                                                      groups=n_intermed, ks=kernel_size,
                                                      stride=stride, norm_type=norm_type,
                                                      act_cls=act_cls, **kwargs)

        # squeeze and excitation layer, if desired
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        if self.has_se:
            num_squeezed_channels = max(1, int(n_inp * se_ratio))
            self.squeeze_expand = nn.Sequential(
                ConvLayerDynamicPadding(ni=n_intermed, nf=num_squeezed_channels, ks=1,
                                        act_cls=act_cls, norm_type=None, **kwargs),
                ConvLayerDynamicPadding(ni=num_squeezed_channels, nf=n_intermed, ks=1,
                                        act_cls=None, norm_type=None,**kwargs))

        # pointwise convolution phase
        self.project_conv = ConvLayerDynamicPadding(ni=n_intermed, nf=n_out, ks=1,
                                                    act_cls = None, **kwargs)
        self.drop_conncet = DropConnect(drop_connect_rate)

    def forward(self, x):
        if self.id_skip: inputs = x # save input only if skip connection

        # expansion
        if self.expand_ratio != 1: x = self.expand_conv(x)

        # depthwise convolution
        x = self.depthwise_conv(x)

        # squeeze and excitation (self attention)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self.squeeze_expand(x_squeezed)
            x = x * x_squeezed.sigmoid() # inplace saves a bit of memory

        # pointwise convolution
        x = self.project_conv(x)

        # skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.n_inp == self.n_out:
            x = self.drop_conncet(x) + inputs  # skip connection
        return x

# Cell
class EfficientNet(nn.Sequential):
    """
    EfficientNet implementation into fastai based on
    https://arxiv.org/abs/1905.11946 and the PyTorch
    implementation of lukemelas (GitHub username)
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    # block arguments remain constant for each model version
    block_arguments =  pd.DataFrame({'num_repeat': [1,2,2,3,3,4,1],
                                     'kernel_size': [3,3,4,3,4,4,3],
                                     'stride':[1,2,2,2,1,2,1],
                                     'expand_ratio':[1,6,6,6,6,6,6,],
                                     'in_channels':[32,16,24,40,80,112,192],
                                     'out_channels':[16,24,40,80,112,192,320],
                                     'se_ratio':[0.25]*7,
                                     'id_skip':[True]*7})

    # calling Efficientnet() without any parementers will default to efficientnet_b0
    def __init__(self,
                 ni=3, # number of input channels
                 num_classes=101, # number of classes
                 width_coefficient=1.0, # width mutliplier
                 depth_coefficient=1.0, # depth multiplier
                 dropout_rate=0.2, # percentage of units to drop
                 drop_connect_rate=0.2, # percentage of inputs to drop
                 depth_divisor=8,
                 min_depth=None, # min depth of the different blocks
                 act_cls = nn.SiLU, # type of activation function, default is Swish (=nn.SiLU)
                 norm_type=NormType.Batch, # type of normalization layer, default is BatchNorm
                ):
        layers = []

        # Stem
        nf_stem = self.get_n_channels(32, width_coefficient, depth_divisor, min_depth)  # number of output channels
        stem = ConvLayerDynamicPadding(ni=ni, nf=nf_stem, ks=3, stride=2,
                                       bias=False, act_cls=None, norm_type=norm_type)
        layers.append(stem)

        # body
        ## build body layer-by-layer
        for idx, row in self.block_arguments.iterrows():
            num_repeat, ks, stride, expand_ratio, ni, nf, se_ratio, id_skip = row
            ni=self.get_n_channels(ni, width_coefficient, depth_divisor, min_depth)
            nf=self.get_n_channels(nf, width_coefficient, depth_divisor, min_depth)
            if depth_coefficient: num_repeat = int(math.ceil(depth_coefficient * num_repeat))

            conv_block = []
            for _ in range(num_repeat):
                conv_block.append(
                    MBConvBlock(n_inp=ni, n_out=nf, kernel_size=ks, stride=stride, se_ratio=se_ratio,
                                id_skip=id_skip, expand_ratio=expand_ratio,
                                drop_connect_rate=drop_connect_rate * float(idx) / len(self.block_arguments), # scale drop connect_rate
                                act_cls=act_cls, norm_type=norm_type
                               ))
                ni, stride = nf, 1 # modify ni and stride if multiple block get stacked
            layers.append(nn.Sequential(*conv_block))

        ## add last ConvLayer of body
        ni = nf # output of final block
        nf = self.get_n_channels(ni, width_coefficient, depth_divisor, min_depth)
        layers.append(ConvLayerDynamicPadding(ni, nf, ks=1, bias=False, act_cls=act_cls, norm_type=norm_type))
        # Head
        head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(nf, num_classes)
        )

        layers.append(head)
        super().__init__(*layers)

    def get_n_channels(self, n_channels,  width_coefficient, depth_divisor, min_depth):
        "calculate number of channels based on width_coefficient, depth_divisor and min_depth and round"
        if not width_coefficient: return filters

        n_channels *= width_coefficient
        min_depth = min_depth or depth_divisor # pay attention to this line when using min_depth
        # follow the formula transferred from official TensorFlow implementation
        new_channels = max(min_depth, int(n_channels + depth_divisor / 2) // depth_divisor * depth_divisor)
        if new_channels < 0.9 * n_channels: # prevent rounding by more than 10%
            new_channels += depth_divisor
        return int(new_channels)


# Cell
_model_urls = {
           'efficientnet_b0': 'https://rad-ai.charite.de/pretrained_models/efficientnet_b0.pth',
           'efficientnet_b1': 'https://rad-ai.charite.de/pretrained_models/efficientnet_b1.pth',
           'efficientnet_b2': 'https://rad-ai.charite.de/pretrained_models/efficientnet_b2.pth',
          }

# Cell
def _efficientnet(arch, width_coefficient, depth_coefficient, dropout_rate, pretrained, progress, **kwargs):
    # arch is currently not used, but will be needed when we can provide pretrained versions.
    model = EfficientNet(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                         dropout_rate=dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(_model_urls[arch],
                                              progress=True)
        model.load_state_dict(state_dict['model'])
    return model

# Cell

def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    return _efficientnet('efficientnet_b0', width_coefficient=1.0, depth_coefficient=1.0,
                         dropout_rate=0.2, pretrained=pretrained, progress=progress, **kwargs)

def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    return _efficientnet('efficientnet_b1', width_coefficient=1.0, depth_coefficient=1.1,
                         dropout_rate=0.2, pretrained=pretrained, progress=progress, **kwargs)

def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    return _efficientnet('efficientnet_b2', width_coefficient=1.1, depth_coefficient=1.2,
                         dropout_rate=0.3, pretrained=pretrained, progress=progress, **kwargs)

def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b3`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b3', width_coefficient=1.2, depth_coefficient=1.4,
                         dropout_rate=0.3, pretrained=False, progress=False, **kwargs)

def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b4`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b4', width_coefficient=1.4, depth_coefficient=2.2,
                         dropout_rate=0.4, pretrained=False, progress=False, **kwargs)

def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b5`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b5', width_coefficient=1.6, depth_coefficient=2.2,
                         dropout_rate=0.4, pretrained=False, progress=False, **kwargs)

def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b6`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b6', width_coefficient=1.8, depth_coefficient=2.6,
                         dropout_rate=0.5, pretrained=False, progress=False, **kwargs)

def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b7`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b7', width_coefficient=2.0, depth_coefficient=3.1,
                         dropout_rate=0.5, pretrained=False, progress=False, **kwargs)

def efficientnet_b8(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_b8`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_b8', width_coefficient=2.2, depth_coefficient=3.6,
                         dropout_rate=0.5, pretrained=False, progress=False, **kwargs)

def efficientnet_l2(pretrained=False, progress=True, **kwargs):
    "load efficientnet with specific scaling coefficients"
    if pretrained: warn('Currently there is no pretrained version available for `efficientnet_l2`. Will load randomly intilialized weights.')
    return _efficientnet('efficientnet_l2', width_coefficient=4.3, depth_coefficient=5.3,
                         dropout_rate=0.5, pretrained=False, progress=False, **kwargs)