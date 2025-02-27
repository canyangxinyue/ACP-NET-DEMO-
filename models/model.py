# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
from typing import Tuple
from addict import Dict

from torch import nn, tensor
from torch import Tensor
import torch
import torch.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'
        self.save_neck=False
        self.neck_out=torch.zeros(1)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out:Tuple[Tensor, Tensor, Tensor, Tensor] = self.backbone(x)
        neck_out = self.neck(backbone_out)
        if self.save_neck:
            self.neck_out=neck_out
        y = self.head(neck_out)

        if isinstance(y,Tensor):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        elif isinstance(y,list):
            y = [F.interpolate(yt, size=(H, W), mode='bilinear', align_corners=True) for yt in y]
        else :
            raise NotImplementedError
        return y



if __name__ == '__main__':
    import torch

    device = torch.device('cuda')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'resnest50', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = Model(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model.name)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
