
import os
import argparse
from veri import verificate
from data_preparing import prepare_test
import ipdb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='closed', help='train/closed/open')
    parser.add_argument('--data_process', type=int, default=1, help='Do the preprocessing when first time for testing.')
    args = parser.parse_args()
    '''
    model path:
    baseline: pretrained_models/0412-0642/backbone_2232.pth
    Variant 1: pretrained_models/0414-1757/backbone_2232.pth
    Variant 2: pretrained_models/0414-1916/backbone_1364.pth
    Variant 3: pretrained_models/0415-1222/backbone_16616.pth
    '''
    test_model_path = 'pretrained_models/0414-1916/backbone_1364.pth'

    if args.mode == 'closed':
        label_path = 'data/test/closed_set/labels.txt'
        pair_path = 'closed_test_pairs.txt'

        img_path = 'data/test/closed_set/test_pairs_align/test_pairs'
        
        #prepare the aligned data for the first time testing
        if args.data_process != 0:
            prepare_test(mode=args.mode)
        acc = verificate(label_path, pair_path, img_path, test_model_path, mode=args.mode) 

    elif args.mode == 'open':
        label_path = 'data/test/open_set/labels.txt'
        pair_path = 'open_test_pairs.txt'

        img_path = 'data/test/open_set/test_pairs_align/test_pairs'
        
        if args.data_process != 0:
            prepare_test(mode=args.mode)

        acc = verificate(label_path, pair_path, img_path, test_model_path, mode=args.mode) 

    elif args.mode == 'train':
        os.system('python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=140.109.135.76 --master_port=1234 train.py')
