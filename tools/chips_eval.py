from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map

def parse_args():
    parser = ArgumentParser(description='Chips Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    return args


def chips_eval(result_file, dataset, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    gt_masks = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        masks = ann['masks']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            # masks = np.vstack([bboxes, ann['masks_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        gt_masks.append(masks)
    if not gt_ignore:
        gt_ignore = None
    print(len(det_results[1][1]))
    print(det_results[1][1])
    dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        # gt_masks,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)

def main():
    args = parse_args()
    
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    chips_eval(args.result, test_dataset, args.iou_thr)


if __name__ == '__main__':
    main()
