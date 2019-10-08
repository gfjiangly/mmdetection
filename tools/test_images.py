# -*- encoding:utf-8 -*-
# @Time    : 2019/9/18 12:09
# @Author  : gfjiang
# @Site    : 
# @File    : test_images.py
# @Software: PyCharm
import os.path as osp
import mmcv
import numpy as np
import cvtools

from mmdet.ops import nms
from mmdet.apis import init_detector, inference_detector


class Detection(object):

    def __init__(self, config, pth):
        self.imgs = []
        self.cfg = config
        self.pth = pth
        print('loading model {} ...'.format(pth))
        self.model = init_detector(self.cfg, self.pth, device='cuda:0')
        self.results = []
        self.img_detected = []

    def __call__(self,
                 imgs_or_path,
                 det_thrs=0.5,
                 vis=False,
                 vis_thr=0.5,
                 save_root=''):
        if isinstance(imgs_or_path, str):
            self.imgs += cvtools.get_files_list(imgs_or_path)
        else:
            self.imgs += imgs_or_path
        prog_bar = mmcv.ProgressBar(len(self.imgs))
        for i, img in enumerate(self.imgs):
            self.detect(img, det_thrs=det_thrs, vis=vis,
                        vis_thr=vis_thr, save_root=save_root)
            prog_bar.update()

    def detect(self,
               img,
               det_thrs=0.5,
               vis=False,
               vis_thr=0.5,
               save_root=''):
        result = inference_detector(self.model, img)
        result = self.nms(result)
        if isinstance(det_thrs, str):
            det_thrs = det_thrs * len(result)
        if vis:
            to_file = osp.join(save_root, osp.basename(img))
            self.vis(img, result, vis_thr=vis_thr, to_file=to_file)
        result = [det[det[..., 4] >= det_thr] for det, det_thr
                  in zip(result, det_thrs)]
        self.img_detected.append(img)
        self.results.append(result)
        return result

    def nms(self, result, nms_th=0.3):
        dets_num = [len(det_cls) for det_cls in result]
        result = np.vstack(result)
        _, ids = nms(result, nms_th)
        total_num = 0
        nms_result = []
        for num in dets_num:
            ids_cls = ids[np.where((total_num <= ids) & (ids < num))[0]]
            nms_result.append(result[ids_cls])
            total_num += num
        return nms_result

    def vis(self, img, bbox_result, vis_thr=0.5,
            to_file='vis.jpg'):
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > vis_thr)[0]
        bboxes = bboxes[inds]
        labels = labels[inds]
        texts = [self.model.CLASSES[index]+'|'+str(round(bbox[4], 2))
                 for index, bbox in zip(labels, bboxes)]
        img = cvtools.draw_boxes_texts(img, bboxes, texts)
        cvtools.imwrite(img, to_file)

    def save_results(self, save):
        mmcv.dump(self.results, save)
