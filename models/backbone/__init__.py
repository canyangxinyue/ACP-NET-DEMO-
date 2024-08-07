# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun

from .resnet import *
from .resnest import *
from .ACP import AcpNet 




__all__ = ['build_backbone']

support_backbone = ['resnet18', 'shuffle_resnet18', 'deformable_resnet18', 'deformable_resnet50',
                    'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'resnest50', 'resnest101', 'resnest200', 'resnest269',
                    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                    'MobileNetV3','u_resnet18','swin_t','AcpNet', 'bisenet','ghost_net','ghost_resnet18','mobile_resnet18',
                    'resnet18_lka','resnet50_lka','resnet18_ska','resnet18_dlka', 'vgg16']


def build_backbone(backbone_name, **kwargs):
    print(backbone_name)
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
