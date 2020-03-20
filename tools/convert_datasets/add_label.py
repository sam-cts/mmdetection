import json
import os.path as osp
import argparse
import sys, glob
# ROOT_DIR = ops.abspath('../../')
def parse_args():
    parser = argparse.ArgumentParser(description='Clean fname')
    parser.add_argument('--dataroot', 
        help='train config file path',
        default='data/20190702_test_chips/')

    parser.add_argument('--siffix', 
        help='annotation file name', 
        default='_data_annotation.json')

    args = parser.parse_args()
    return args

def main():
    ROOT_DIR = osp.abspath('./')
    args = parse_args()
    sets = ['train', 'val', 'test']
    dataroot = osp.join(ROOT_DIR, args.dataroot)
    for i in range(len(sets)):
        print(f'Cleaning {sets[i]}')
        dataset = sets[i]
        fpath = osp.join(dataroot, f'annotations/{dataset}{args.siffix}')
        ofpath = osp.join(dataroot, f'annotations/old_2_{dataset}{args.siffix}')
        ann_dict = json.load(open(fpath, 'r'))
        new_ann_dict = {}
        for i, p in ann_dict.items():
            for r in p['regions']:
                r['label'] = 'chips'
            new_ann_dict[p['filename']] = p
        # write data
        json.dump(ann_dict, open(ofpath, 'w'))
        json.dump(new_ann_dict, open(fpath, 'w'))

if __name__ == '__main__':
    main()
