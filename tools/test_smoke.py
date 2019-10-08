# -*- encoding:utf-8 -*-
# @Time    : 2019/9/14 16:52
# @Author  : gfjiang
# @Site    : 
# @File    : test_hat.py
# @Software: PyCharm
import os.path as osp

import test_images as test


class SmokingDetection(test.Detection):

    def save_results(self, save):
        str_results = ''
        for i, img in enumerate(self.img_detected):
            result = self.results[i]
            img = osp.basename(img)
            for dets in result:
                for box in dets:
                    bbox_str = ','.join(map(str, map(int, box[:4])))
                    str_results += ' '.join([img, bbox_str]) + '\n'
        with open(save, 'w') as f:
            f.write(str_results)


if __name__ == '__main__':
    config_file = '../configs/smoke/scale_cascade_rcnn_r50_fpn_2x.py'
    pth_file = 'work_dirs/cascade_rcnn_r50_fpn_2x/scale_smoke/epoch_24.pth'
    smoke_det = SmokingDetection(config_file, pth_file)
    smoke_det('/media/gfjiang/办公/data/smoke/V1.0/test', det_thrs=[0.5],
              vis=True, vis_thr=0.1, save_root='work_dirs/smoke_vis')
    smoke_det.save_results('work_dirs/cascade_rcnn_r50_fpn_2x/scale_smoke/smoke_result.txt')
