# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/8/25 19:58
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import argparse
import os.path as osp
from tqdm import tqdm
import mmcv
import cv2
import numpy as np
import pycocotools.mask as maskUtils
from collections import defaultdict
from multiprocessing import Pool
import cvtools

from mmdet.ops import nms
from mmdet.core import coco_eval, results2json
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test images detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('results', help='results file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--out',
                        default='work_dirs/',
                        help='results path')
    args = parser.parse_args()
    return args


def convert_mask_to_rbbox(result, score_thr=0.1):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    rbboxes = []
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            yx_mask = np.where(mask)
            mask_coors = np.vstack((yx_mask[1], yx_mask[0])).transpose()
            rbbox = cv2.minAreaRect(mask_coors)
            rect = cvtools.xywh_to_x1y1x2y2([rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1]])
            x1y1x2y2x3y3x4y4 = cvtools.rotate_rect(rect, rbbox[0], rbbox[2])
            rbboxes.append(x1y1x2y2x3y3x4y4)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
        bboxes = bboxes[inds]
        labels = labels[inds]
    scores = bboxes[:, -1]
    bboxes = bboxes[:, :-1]
    rbboxes = np.array(rbboxes).astype(np.int)
    return labels, scores, bboxes, rbboxes


def crop_bbox_map_back(bb, crop_start):
    bb_shape = bb.shape
    original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
    return original_bb.reshape(bb_shape)


def genereteImgResults(anns, results):
    imgResults = defaultdict(list)
    for ann, result in tqdm(zip(anns, results)):
        labels, scores, bboxes, rbboxes = convert_mask_to_rbbox(result)
        if 'crop' in ann:
            bboxes = crop_bbox_map_back(bboxes, ann['crop'][:2])  # nms需要
            rbboxes = crop_bbox_map_back(rbboxes, ann['crop'][:2])
        assert len(rbboxes) == len(labels)
        if len(labels) > 0:
            result = [bboxes, rbboxes, labels, scores]
            imgResults[ann['filename']].append(result)
    return imgResults


def merge_results(ann_file, result_file, n_worker=0):
    anns = mmcv.load(ann_file)
    results = mmcv.load(result_file)
    print('convert mask to rbbox...')
    imgResults = defaultdict(list)
    if n_worker > 0:
        pool = Pool(processes=n_worker)
        num = len(anns) // n_worker
        anns_group = [anns[i:i + num] for i in range(0, len(anns), num)]
        results_group = [results[i:i + num] for i in range(0, len(results), num)]
        res = []
        for anns, results in tqdm(zip(anns_group, results_group)):
            res.append(pool.apply_async(genereteImgResults, args=(anns, results,)))
        pool.close()
        pool.join()
        for item in res:
            imgResults.update(item.get())
    else:
        imgResults = genereteImgResults(anns, results)
    for filename, result in imgResults.items():
        bboxes = np.vstack([bb[0] for bb in result])
        rbboxes = np.vstack([bb[1] for bb in result])
        labels = np.hstack([bb[2] for bb in result])
        scores = np.hstack([bb[3] for bb in result])
        _, ids = nms(np.hstack([bboxes, scores[:, np.newaxis]]), 0.3)
        # rbboxes = np.hstack([rbboxes, labels, scores])
        imgResults[filename] = [rbboxes[ids], labels[ids], scores[ids]]
    return imgResults


def ImgResults2CatResults(imgResults):
    catResults = defaultdict(list)
    for filename in imgResults:
        rbboxes = imgResults[filename][0]
        cats = imgResults[filename][1]
        scores = imgResults[filename][2]
        for ind, cat in enumerate(cats):
            catResults[cat].append([filename, scores[ind], rbboxes[ind]])
    return catResults


def writeResults2DOTATestFormat(catResults, class_names):
    for cat_id, result in catResults.items():
        lines = []
        for filename, score, rbbox in result:
            filename = osp.splitext(filename)[0]
            bbox = list(map(str, list(rbbox)))
            score = str(round(score, 3))
            lines.append(' '.join([filename] + [score] + bbox))
        cvtools.write_list_to_file(
            lines, osp.join('dota/test', 'Task1_'+class_names[cat_id]+'.txt'))


def evaluate_results_file(args):
    cfg = mmcv.Config.fromfile(args.config)
    datasets = build_dataset(cfg.data.val)
    results = mmcv.load(args.results)
    result_files = results2json(datasets, results, args.out)
    coco_eval(result_files, args.eval, datasets.coco)


def generete_dota_test_results(args):
    cfg = mmcv.Config.fromfile(args.config)
    ann_file = cfg.data.test.ann_file
    imgResults = merge_results(ann_file, args.results, n_worker=4)
    catResults = ImgResults2CatResults(imgResults)
    from mmdet.datasets.dota import DOTADataset
    class_names = DOTADataset.CLASSES
    writeResults2DOTATestFormat(catResults, class_names)


if __name__ == '__main__':
    args = parse_args()
    if args.eval:
        evaluate_results_file(args)
    else:
        generete_dota_test_results(args)


