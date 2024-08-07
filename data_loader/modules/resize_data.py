import random

import cv2
import numpy as np

class ResizeData():
    def __init__(self, size=(640, 640), keep_ratio=True):
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        scale = self.size[0]/h,self.size[1]/w 
        im = cv2.resize(im, dsize=None, fx=scale[1], fy=scale[0])
        if self.keep_ratio:
            # text_polys *= scale
            text_polys[:,:, 0] *= scale[1]
            text_polys[:,:, 1] *= scale[0]
        
        data['shape'] = im.shape[0:2]
        data['img'] = im
        data['text_polys'] = text_polys
        return data


class MakeDividable():
    def __init__(self, divisor=32):
        self.divisor=divisor

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        dh, dw = self.divisor - h % self.divisor , self.divisor- w % self.divisor       
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))  # add border

        for text_poly in text_polys:
            text_poly[:,0]+= left
            text_poly[:,1]+= top

        data['shape'] = im.shape[0:2]
        data['img'] = im
        data['text_polys'] = text_polys
        return data
    
class LimitSize():
    def __init__(self,max_size):
        self.max_size=max_size
        
    def __call__(self, data: dict) -> dict:
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        long_size=max(h,w)
        scale=1
        
        while long_size*scale > self.max_size:
            scale/=2
        
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)

        for text_poly in text_polys:
            text_poly[:,0]*=scale
            text_poly[:,1]*=scale

        data['shape'] = im.shape[0:2]
        data['img'] = im
        data['text_polys'] = text_polys
        return data