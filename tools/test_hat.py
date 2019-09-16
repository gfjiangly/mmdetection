# -*- encoding:utf-8 -*-
# @Time    : 2019/9/14 16:52
# @Author  : gfjiang
# @Site    : 
# @File    : test_hat.py
# @Software: PyCharm
import os.path as osp
import mmcv
import numpy as np
import cvtools

from mmdet.ops import nms
from mmdet.apis import init_detector, inference_detector


class HatDetection(object):

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
        result = [det[det[..., 4] > det_thr] for det, det_thr
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
        str_results = ''
        for i, img in enumerate(self.img_detected):
            result = self.results[i]
            img = osp.basename(img)
            for cls_index, dets in enumerate(result):
                cls = self.model.CLASSES[cls_index]
                for box in dets:
                    bbox_str = ','.join(map(str, map(int, box[:4])))
                    str_results += ' '.join([img, cls, bbox_str]) + '\n'
        with open(save, 'w') as f:
            f.write(str_results)


if __name__ == '__main__':
    config_file = '../configs/hat_detect/hatv1_cascade_rcnn_r50_fpn_1x.py'
    pth_file = 'work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/hat_detect/epoch_12.pth'
    hat_det = HatDetection(config_file, pth_file)
    hat_det('/media/gfjiang/办公/data/hat_V1.0/test', det_thrs=[0.68, 0.79],
            vis=True, vis_thr=0.1, save_root='work_dirs/hat_vis')
    hat_det.save_results('work_dirs/hatv1_cascade_rcnn_r50_fpn_1x/hat_detect/hatdetect_result.txt')
