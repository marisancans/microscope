import torch.nn as nn
import torchvision.models as models



class VGG16(nn.Module):

    r"""A module that augments VGG16 as defined in 'Very Deep Convolutional
    Networks for Large-Scale Image Recognition'
    (https://arxiv.org/pdf/1409.1556.pdf).

    1. It can return first `n_layers` of the VGG16.
    2. It can add coordinates to feature maps prior to each convolution.
    3. It can return all outputs (including intermediate outputs) of the
        VGG16.

    Args:
        n_layers (int): Use first `n_layers` layers of the VGG16
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        return_intermediate_outputs (bool, optional): If `True`, return
            outputs of the each layer in the VGG16 as a list otherwise
            return output of the last layer of first `n_layers` layers
            of the VGG16. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: Output of the last layer of the selected subpart of VGG16
            or the list that contains outputs of the each layer depending on
            `return_intermediate_outputs`

    Examples:
        >>> vgg16 = VGG16(16, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = VGG16(16, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, n_layers, pretrained=True, use_coordinates=False,
                 return_intermediate_outputs=False, usegpu=True):
        super(VGG16, self).__init__()

        self.use_coordinates = use_coordinates
        self.return_intermediate_outputs = return_intermediate_outputs

        self.cnn = models.__dict__['vgg16'](pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        modules = list(self.cnn.children())
        modules[0] = nn.Conv2d(1, 64, 3, 1, 1)
        self.cnn = nn.Sequential(*modules)
        
        self.cnn = nn.Sequential(*list(self.cnn.children())[: n_layers])

        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, True, usegpu)

    def __get_outputs(self, x):
        if self.use_coordinates:
            return self.cnn(x)

        outputs = []
        for i, layer in enumerate(self.cnn.children()):
            x = layer(x)
            outputs.append(x)

        return outputs

    def forward(self, x):
        outputs = self.__get_outputs(x)

        if self.return_intermediate_outputs:
            return outputs

        return outputs[-1]


class SkipVGG16(nn.Module):

    r"""A module that returns output of 7th convolutional layer of the
    VGG16 along with outputs of the 2nd and 4th convolutional layers.

    Args:
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: List of outputs of the 2nd, 4th and 7th convolutional
            layers of the VGG16, respectively.

    Examples:
        >>> vgg16 = SkipVGG16(True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = SkipVGG16(True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, pretrained=True, use_coordinates=False,
                 usegpu=True):
        super(SkipVGG16, self).__init__()

        self.use_coordinates = use_coordinates

        self.outputs = [3, 8]
        self.n_filters = [64, 128]

        self.model = VGG16(n_layers=16, pretrained=pretrained,
                           use_coordinates=self.use_coordinates,
                           return_intermediate_outputs=True,
                           usegpu=usegpu)

    def forward(self, x):

        if self.use_coordinates:
            outs = self.model(x)
            out = [o for i, o in enumerate(outs) if i in self.outputs]
            out.append(outs[-1])
        else:
            out = []
            for i, layer in enumerate(list(self.model.children())[0]):
                x = layer(x)
                if i in self.outputs:
                    out.append(x)
            out.append(x)

        return out
