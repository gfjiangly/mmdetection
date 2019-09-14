# -*- encoding:utf-8 -*-
# @Time    : 2019/8/24 15:40
# @Author  : gfjiang
# @Site    : 
# @File    : dota.py
# @Software: PyCharm
import mmcv

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class HatDataset(CocoDataset):

    CLASSES = ('head', 'hat')

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
