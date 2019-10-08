# -*- encoding:utf-8 -*-
# @Time    : 2019/9/19 20:36
# @Author  : gfjiang
# @Site    : 
# @File    : extract_weight.py
# @Software: PyCharm
import torch


def extract_weights(model, cls_num=21):
    model_coco = torch.load(model)

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(cls_num, 1024)
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(cls_num, 1024)
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(cls_num, 1024)
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(cls_num)
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(cls_num)
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(cls_num)
    # save new model
    torch.save(model_coco, model+"_cls_{}.pth".format(cls_num))


