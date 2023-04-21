import torch.nn as nn
from models.backbone import resnet
from models.backbone.hrnet import hrnet18
from models.backbone.vgg import vgg16_bn
from models.backbone.mobilenetv2 import mobilenetv2
from models.backbone.swin import swin_tiny

def build_backbone(args, backbone, in_channels):
    if backbone == 'hrnet18':
        print("****************backbone is HRNet18****************")
        return hrnet18(args,in_channels)
    elif backbone == 'resnet18':
        print("****************backbone is ResNet18****************")
        return resnet.ResNet18(args, in_channels)
    elif backbone == 'resnet50':
        print("****************backbone is ResNet50****************")
        return resnet.ResNet50(args, in_channels)
    elif backbone == "vgg16":
        print("****************backbone is vgg16_bn****************")
        return vgg.vgg16_bn(args,in_channels)
    elif backbone == "mobilenetv2":
        print("****************backbone is mobilenetv2_1.0****************")
        return mobilenetv2(args,in_channels)
    elif backbone == "swint":
        print("****************backbone is swin_tiny****************")
        return swin_tiny(args,in_channels)
    else:
        raise NotImplementedError
