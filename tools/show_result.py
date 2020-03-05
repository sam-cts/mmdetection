import os.path as ops
import argparse
import glob
from mmdet.apis import init_detector, inference_detector, show_result
# from utils.viz import show_result
# from mmdet.core import coco_eval, results2json, wrap_fp16_model
# from mmdet.datasets import build_dataloader, build_dataset
# from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Image Inference Result Display')
    parser.add_argument(
        '--config',
        help='config file',
        type=str)
    parser.add_argument(
        '--model',
        help='corresponse model trained with the config file',
        type=str)
    parser.add_argument(
        '--imgspath',
        help='path to the folder contain images to inference',
        type=str)
    parser.add_argument(
        '--output',
        help='path to output folder',
        type=str,  
        default='output/')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Initialize the inference config and model
    model = init_detector(args.config, args.model, device='cuda:0')

    img_dir = args.imgspath
    img_list = glob.glob1(ops.abspath(img_dir), '*.jpg')

    for img in img_list:
        img_path = ops.join(img_dir, img)
        result = inference_detector(model, img_path)
        outimg = ops.join(args.output, img)
        # print(type(model.CLASSES))
        
        show_result(img_path, result, [model.CLASSES], show=False, score_thr=0.8, out_file=outimg)#, show_mask=True)
        

if __name__ == '__main__':
    main()