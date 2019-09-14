from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map
from mmdet.core.evaluation.coco_utils import eval_recalls


def voc_eval(result_file, dataset, iou_thr=0.5,
             eval_type='bbox', max_dets=(100, 300, 1000)):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    if eval_type == 'bbox':
        eval_map(
            det_results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=iou_thr,
            dataset=dataset_name,
            print_summary=True)
    elif eval_type == 'proposal_fast':
        eval_recalls(
            gt_bboxes,
            det_results,
            np.array(max_dets),
            iou_thr,
            print_summary=True)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',     # support proposal_fast
        help='eval types')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr, args.eval)


if __name__ == '__main__':
    main()
