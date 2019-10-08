# -*- encoding:utf-8 -*-
# @Time    : 2019/9/18 20:17
# @Author  : gfjiang
# @Site    : 
# @File    : review.py
# @Software: PyCharm
import mmcv

from mmdet.core import coco_eval, results2json
from mmdet.datasets import build_dataset
from tools.val_test import ValTest


def review_model():
    args = [
        # ('../configs/review/hat/hatv1_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/hat/epoch_12.pth',
        #  './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/hat/val_results.pkl'),
        #
        # ('../configs/review/hat/hatv1_cascade_rcnn_x101_32x4d_fpn_1x.py',
        #  './work_dirs/hatv1_cascade_rcnn_x101_32x4d_fpn_1x/hat/epoch_12.pth',
        #  './work_dirs/hatv1_cascade_rcnn_x101_32x4d_fpn_1x/hat/val_results.pkl'),
        #
        # ('../configs/review/hat/hatv1_scale_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/scale_hat/epoch_12.pth',
        #  './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/scale_hat/val_results.pkl')

        # ('../configs/review/hat/hatv1_scale_cascade_rcnn_r50_fpn_2x.py',
        #  '/home/gfjiang/newdisk/model_file/mmdet/hatv1_cascade_rcnn_r50_fpn_2x/scale_hat/epoch_24.pth',
        #  '/home/gfjiang/newdisk/model_file/mmdet/hatv1_cascade_rcnn_r50_fpn_2x/scale_hat/test_shwd_results.pkl')

        # ('../configs/hat/hatv2_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/baseline/epoch_12.pth',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/baseline/hatv2_val_results.pkl'),
        #
        # ('../configs/hat/hatv2_color_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/color/epoch_12.pth',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/color/hatv2_val_results.pkl'),
        #
        # ('../configs/hat/hatv2_bright_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/bright/epoch_12.pth',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/bright/hatv2_val_results.pkl'),

        ('../configs/hat/hatv2_color_cascade_rcnn_r50_fpn_2x.py',
         './work_dirs/hatv2_cascade_rcnn_r50_fpn_2x/color/epoch_24.pth',
         './work_dirs/hatv2_cascade_rcnn_r50_fpn_2x/color/hatv2_val_results.pkl'),

        ('../configs/hat/hatv2_bright_cascade_rcnn_r50_fpn_2x.py',
         './work_dirs/hatv2_cascade_rcnn_r50_fpn_2x/bright/epoch_24.pth',
         './work_dirs/hatv2_cascade_rcnn_r50_fpn_2x/bright/hatv2_val_results.pkl')

        # ('../configs/hat/hatv2_nopretrain_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/nopretrain/epoch_12.pth',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/nopretrain/hatv2_val_results.pkl'),
        #
        # ('../configs/hat/hatv2_cocopretrain_cascade_rcnn_r50_fpn_1x.py',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/cocopretrain/epoch_12.pth',
        #  './work_dirs/hatv2_cascade_rcnn_r50_fpn_1x/cocopretrain/hatv2_val_results.pkl')

    ]
    eval = ['bbox']
    for arg in args:
        config, checkpoint, out = arg
        val = ValTest(config, checkpoint, eval, out=out)
        val.detect()


def review_results():
    args = [('../configs/review/hat/hatv1_cascade_rcnn_r50_fpn_1x.py',
             './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/hat/val_results.pkl'),

            ('../configs/review/hat/hatv1_cascade_rcnn_x101_32x4d_fpn_1x.py',
             './work_dirs/hatv1_cascade_rcnn_x101_32x4d_fpn_1x/hat/val_results.pkl'),

            ('../configs/review/hat/hatv1_scale_cascade_rcnn_r50_fpn_1x.py',
             './work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/scale_hat/val_results.pkl')]
    eval = ['bbox']
    for arg in args:
        config, results_file = arg
        cfg = mmcv.Config.fromfile(config)
        datasets = build_dataset(cfg.data.val)
        results = mmcv.load(results_file)
        result_files = results2json(datasets, results, results_file)
        coco_eval(result_files, eval, datasets.coco)


if __name__ == '__main__':
    review_model()
