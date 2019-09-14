# -*- encoding:utf-8 -*-
# @Time    : 2019/8/24 15:40
# @Author  : gfjiang
# @Site    : 
# @File    : dota.py
# @Software: PyCharm
import os.path as osp
import mmcv
import numpy as np
import warnings
from shutil import copyfile
import asyncio
import torch

from .coco import CocoDataset
from .registry import DATASETS, PIPELINES
from .pipelines.loading import LoadImageFromFile, LoadAnnotations


async def save_image(img, filename):
    mmcv.imwrite(img, filename)


async def load_image(filename):
    return mmcv.imread(filename)


@DATASETS.register_module
class DOTADataset(CocoDataset):

    CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 'ship',
               'soccer-ball-field', 'basketball-court', 'ground-track-field', 'small-vehicle',
               'harbor', 'baseball-diamond', 'tennis-court', 'roundabout', 'storage-tank')

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
class DOTALoadImageFromFile(LoadImageFromFile):

    def __call__(self, results):
        if 'crop' not in results['img_info']:
            return super().__call__(results)
        img_info = results['img_info']
        filename = osp.splitext(osp.basename(img_info['filename']))[0]
        suffix = osp.splitext(osp.basename(img_info['filename']))[1]
        crop_str = list(map(str, img_info['crop']))
        # crop_filename = osp.join(results['img_prefix'],
        #                          '../../crop',
        #                          '_'.join([filename] + crop_str) + suffix)
        crop_filename = osp.join('/media/gfjiang/办公/data/DOTA/crop',
                                 '_'.join([filename] + crop_str) + suffix)
        if not osp.isfile(crop_filename):
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
            img = mmcv.imread(filename)
            sx, sy, ex, ey = img_info['crop']
            img = img[sy:ey, sx:ex]
            # mmcv.imwrite(img, crop_filename)
            coroutine = save_image(img, crop_filename)
            task = asyncio.ensure_future(coroutine)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(task)
        else:
            img = mmcv.imread(crop_filename)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = crop_filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


@PIPELINES.register_module
class DOTALoadAnnotations(LoadAnnotations):

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        # filter the image that greater 200 boxes to save GPU memory
        if len(results['gt_bboxes']) > 200:
            warnings.warn(
                'Skip the image "{}" that gt bboxes have {}'.format(
                    results['filename'], len(results['gt_bboxes'])))
            copyfile(results['filename'], '../data/ignored/'+osp.basename(results['filename']))
            return None
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results
