# -*- encoding:utf-8 -*-
# @Time    : 2019/9/16 23:09
# @Author  : gfjiang
# @Site    : 
# @File    : smoke.py
# @Software: PyCharm
import mmcv
import os.path as osp
import numpy as np
from PIL import Image

from .coco import CocoDataset
from .registry import DATASETS, PIPELINES
from .pipelines.loading import LoadImageFromFile


@DATASETS.register_module
class SmokeDataset(CocoDataset):

    CLASSES = ('smoke',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 mode='train'):
        self.mode = mode
        super().__init__(ann_file, pipeline, data_root, img_prefix,
                         seg_prefix, proposal_file, test_mode)

    def load_annotations(self, ann_file):
        if self.mode == 'test':
            return mmcv.load(ann_file)
        else:
            return super().load_annotations(ann_file)


@PIPELINES.register_module
class SmokeLoadImageFromFile(LoadImageFromFile):

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        if img is None:     # opencv problem for same images
            img = np.array(Image.open(filename))
            if img is None:
                print('image {} is None'.format(filename))
                return None
            if len(img.shape) == 2:
                img = np.repeat(img[..., np.newaxis], 3, axis=2)
            else:
                img = img[..., ::-1]
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
