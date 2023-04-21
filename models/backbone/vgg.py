import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.conv = nn.Conv2d(270,3,kernel_size=1,stride=1,bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        self.channels = [64,128,256,512]

    def forward(self, x):

        #print(self.features)

        #x = self.conv(x)

        x0 = self.features[:7](x)
        x1 = self.features[7:14](x0)
        x2 = self.features[14:23](x1)
        x3= self.features[24:33](x2)
        #x4 = self.features[34:43](x3)

        print(x.shape)
        print(x0.shape)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        # print(x4.shape)

        return [x0,x1,x2,x3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(output_stride, BatchNorm, pretrained=True):
    """VGG 16-layer models (configuration "D")

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), output_stride, BatchNorm)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(args, in_channels, pretrained=True, **kwargs):
    """VGG 16-layer models (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], in_channels, batch_norm=True), **kwargs)
    if pretrained:
        model_path = '../../pretrain_model/vgg16_bn-6c64b313.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        model_dict = {}

        if 'WHUHi' in args.dataset:
            state_dict = model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict and 'features.0' not in k:
                    model_dict[k] = v
            state_dict.update(model_dict)

            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(torch.load(model_path), strict=False)

    return model

