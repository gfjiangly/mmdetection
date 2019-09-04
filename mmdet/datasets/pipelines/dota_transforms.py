# -*- encoding:utf-8 -*-
# @Time    : 2019/9/4 20:29
# @Author  : gfjiang
# @Site    : 
# @File    : custom_transforms.py
# @Software: PyCharm
import mmcv
import numpy as np
from imagecorruptions import corrupt
from numpy import random

from ..registry import PIPELINES
from .transforms import RandomCrop, Expand


@PIPELINES.register_module
class DOTARandomCrop(RandomCrop):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __call__(self, results):
        if random.randint(2):
            return results

        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i, is_valid in enumerate(valid_inds):
                    if is_valid:
                        gt_mask = results['gt_masks'][i]
                        valid_gt_masks.append(
                            gt_mask[crop_y1:crop_y2, crop_x1:crop_x2])
                results['gt_masks'] = valid_gt_masks

        return results


@PIPELINES.register_module
class DOTAExpand(Expand):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __call__(self, results):
        if random.randint(2):
            return results

        img, boxes, masks = [
            results[k]
            for k in ('img', 'gt_bboxes', 'gt_masks')
        ]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes += np.tile((left, top), 2)

        expand_masks = []
        for mask in masks:
            expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                  0).astype(img.dtype)
            expand_mask[top:top + h, left:left + w] = mask
            expand_masks.append(expand_mask)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes
        results['gt_masks'] = expand_masks
        return results



