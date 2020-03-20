import numpy as np
import json, os
import skimage.io
import skimage.draw
from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class ChipsDataset(CustomDataset):

    CLASSES = ('chips', )
    
    def __init__(self, **kwargs):
        super(ChipsDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i+1 for i, cat in enumerate(self.CLASSES)}
        self.cat_ids = list(self.CLASSES)
    

    def load_annotations(self, ann_file):
        img_infos = []
        with open(ann_file, 'r') as json_file:
            img_json = json.load(json_file)
        self.img_ids = img_json.keys()
        for img_id in self.img_ids:
            fname = str(img_json[img_id]['filename'])
            img_path = os.path.join(self.img_prefix, fname)
            height, width = skimage.io.imread(img_path).shape[:2]

            if type(img_json[img_id]['regions']) is dict:
                polygons = [r['shape_attributes'] for r in img_json[img_id]['regions'].values()]
                labels = ['chips' for r in img_json[img_id]['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in img_json[img_id]['regions']]
                labels = ['chips' for r in img_json[img_id]['regions']]

            img_infos.append(
                dict(id=img_id, 
                    filename=fname, 
                    width=width, 
                    height=height, 
                    polygons=polygons, 
                    labels= labels))
        return img_infos

    def get_ann_info(self, idx):

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_masks_ann = []

        labels = self.img_infos[idx]['labels']
        polygons = self.img_infos[idx]['polygons']
        height = self.img_infos[idx]['height']
        width = self.img_infos[idx]['width']
        
        for i , p in enumerate(polygons):
            
            # Extract bbox from polygon
            hor = np.sort(p['all_points_x']) 
            ver = np.sort(p['all_points_y'])
            x1 = int(hor[0])
            x2 = int(hor[-1])
            y1 = int(ver[0])
            y2 = int(ver[-1])
            x2 += 1
            y2 += 1
            bbox = (x1, y1, x2, y2)
            gt_bboxes.append(bbox)

            # Extract binary mask
            mask = np.zeros([height, width])
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc]=1

            gt_masks_ann.append(mask.astype(np.uint8))

            # Extract label
            gt_labels.append(self.cat2label[labels[i]])
       

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            gt_labels_ignore = np.array(gt_labels_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            gt_labels_ignore = np.zeros((0, ), dtype=np.float32)         

        seg_map = self.img_infos[idx]['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            labels_ignore=gt_labels_ignore, 
            masks=gt_masks_ann, 
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

