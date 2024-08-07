# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 11:34
# @Author  : zhoujun

from .Empty import Empty

__all__ = ['build_neck']
support_neck = ['Empty']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
