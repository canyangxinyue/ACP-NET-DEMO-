# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import math
import pathlib
import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

import sys
sys.path.append(".")
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from base import BaseDataSet
from utils import order_points_clockwise, order_points_clockwise_list, get_datalist, load,expand_polygon



class CTW1500Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = (np.array(list(map(float, params[:28]))).reshape(-1, 2)).astype(np.float32)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = ",".join(params[28:])
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception as e:
                    print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

class CTW1500TrainDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        import xml.dom.minidom
        dom = xml.dom.minidom.parse(label_path)
        dom_boxes=dom.documentElement.getElementsByTagName('box')
        for dom_box in dom_boxes:
            line=dom_box.getElementsByTagName("segs")[0].firstChild.data
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                box = (np.array(list(map(float, params))).reshape(-1, 2)).astype(np.float32)
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    label = dom_box.getElementsByTagName("label")[0].firstChild.data
                    texts.append(label)
                    ignores.append(label in self.ignore_tags)
            except Exception as e:
                print('load label failed on {}'.format(label_path),e)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data



if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from utils import parse_config, show_img, plt, draw_bbox

    config = anyconfig.load('config/ctw1500_resnet18_FPN_DB_CT_head_polyLR.yaml')
    read_type='train'
    show_type='det_train'
    # dir_name="output/datasets/td500/validate"
    dir_name="fig"
    config = parse_config(config)
    dataset_args = config['dataset'][read_type]['dataset']['args']
    dataset_type = config['dataset'][read_type]['dataset']['type']
    config['dataset'][read_type]['loader']['shuffle']=False
    # config['dataset']['train']['dataset']['args']['filter_keys'].remove('text_polys')
    if 'img_name' in dataset_args['filter_keys']: 
        dataset_args['filter_keys'].remove('img_name')
    # dataset_args.pop('data_path')
    # data_list = [(r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')]
    train_data = eval(dataset_type)(data_path=dataset_args.pop('data_path'), transform=transforms.ToTensor(),
                                  **dataset_args)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    import matplotlib
    matplotlib.use("agg")
    # matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patches 
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    # plt.rcParams['axes.unicode_minus']=False


    os.makedirs(dir_name,exist_ok=True)
    
    if show_type=='det_train':
        from dataset_visualize import save_train_image
        save_image=save_train_image
    elif show_type=='det_validate':
        from dataset_visualize import save_test_image
        save_image=save_test_image
    elif show_type=='rec':
        from dataset_visualize import save_rec_image
        save_image=save_rec_image
    elif show_type=='distance_map':
        from dataset_visualize import save_distance_image
        save_image=save_distance_image
    
    for i, data in enumerate(tqdm(train_loader)):
        save_image(data['img_name'][0],data,output_dir=dir_name)       
        pass
