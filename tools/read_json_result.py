import json
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Examine Result File')
    parser.add_argument('result', help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Read File
    result = json.load(open(args.result, 'r'))

    print(type(result))
    print(len(result))



if __name__ == "__main__":
    main()