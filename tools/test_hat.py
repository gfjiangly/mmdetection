# -*- encoding:utf-8 -*-
# @Time    : 2019/9/14 16:52
# @Author  : gfjiang
# @Site    : 
# @File    : test_hat.py
# @Software: PyCharm
import mmcv
import cvtools

from mmdet.apis import init_detector, inference_detector


class HatDetection(object):

    def __init__(self, config, pth):
        self.imgs = []
        self.cfg = mmcv.Config.fromfile(config)
        self.pth = pth
        print('loading model {} ...'.format(pth))
        self.model = init_detector(config, self.pth, device='cuda:0')
        self.results = []

    def __call__(self, img_path):
        self.imgs += cvtools.get_files_list(img_path)
        for img in self.imgs:
            result = inference_detector(self.model, img)
            self.results.append(result)

    def detect(self, img):
        result = inference_detector(self.model, img)
        self.results.append(result)
        return result

    def save_results(self, save):
        pass


if __name__ == '__main__':
    config_file = '../configs/hat_detect/cascade_rcnn_r50_fpn_1x.py'
    pth_file = 'work_dirs/cascade_rcnn_r50_fpn_1x/hat_detect/epoch_12.pth'
    hat_det = HatDetection(config_file, pth_file)
    hat_det.detect('/media/gfjiang/办公/data/hat_V1.0/JPEGImages/bdaqm_1.jpg')
    hat_det.save_results('work_dirs/cascade_rcnn_r50_fpn_1x/hat_detect/hatdetect_result.txt')
