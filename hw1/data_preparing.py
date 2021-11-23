
import os
import ipdb
import glob
import shutil
import argparse
from data_utils.align_dataset_mtcnn_v1 import mtcnn_align


def alignment(mode='closed'):
    if mode == 'closed':
        input_dir = 'data/test/closed_set/test_pairs'
        output_dir = 'data/test/closed_set/test_pairs_align'
    elif mode == 'open':
        data_root = 'data/test/open_set/test_pairs'
        os.makedirs(os.path.join(data_root, 'test_pairs'), exist_ok=True)
        for f in os.listdir(data_root):

            if not os.path.isdir(os.path.join(data_root,f)):
                src = os.path.join(data_root, f)
                dst = os.path.join(data_root, 'test_pairs', f)
                shutil.move(src, dst)
            
        input_dir = 'data/test/open_set/test_pairs/'
        output_dir = 'data/test/open_set/test_pairs_align/'

    elif mode == 'unlabel':
        data_root = 'data/test/open_set/unlabeled_data'
        os.makedirs(os.path.join(data_root, 'unlabeled_data'), exist_ok=True)
        for f in os.listdir(data_root):

            if not os.path.isdir(os.path.join(data_root,f)):
                src = os.path.join(data_root, f)
                dst = os.path.join(data_root, 'unlabeled_data', f)
                shutil.move(src, dst)
            
        input_dir = 'data/test/open_set/unlabeled_data'
        output_dir = 'data/test/open_set/unlabeled_data_align/'




    image_size = 112
    margin = 44
    random_order = True
    gpu_memory_fraction = 1.0
    detect_multiple_faces = False

    mtcnn_align(input_dir, output_dir, image_size, margin, random_order, gpu_memory_fraction, detect_multiple_faces)  



def prepare_test(mode='closed'):

    # alignment
    alignment(mode)


if __name__ == '__main__':
    prepare_test(mode='unlabel')