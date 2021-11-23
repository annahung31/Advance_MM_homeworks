## Reference: https://github.com/davidsandberg/facenet/blob/master/src/lfw.py


import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import ipdb
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # general
    # for eval: --data-dir ../train/val_split --pair-list val_pairs.txt --output val.bin
    # for closed test: --data-dir ../test/closed_set/test_pairs_align/test_pairs --pair-list test_pairs.txt --output test.bin
    # parser.add_argument('--mode', default='val', help='val, test')
    parser.add_argument('--data-dir', default='../train/val_split', help='')
    parser.add_argument('--pair-list', default='val_pairs.txt', help='')
    parser.add_argument('--image-size', type=str, default='112,112', help='')
    parser.add_argument('--output', default='val.bin', help='path to save.')

    args = parser.parse_args()
    data_dir = args.data_dir
    image_size = [int(x) for x in args.image_size.split(',')]
    lfw_pairs = read_pairs(args.pair_list)
    lfw_paths, issame_list = get_paths(data_dir, lfw_pairs) # or jpg
    lfw_bins = []

    '''
    len(lfw_paths) == len(issame_list)*2
    A pair is recorded as (lfw_paths[i], lfw_paths[i+1]) for i % 2 == 0, 
    and the label of the pair is issame_list[i].
    For example:
    lfw_paths[0] = pair_0_0
    lfw_paths[1] = pair_0_1
    lfw_paths[2] = pair_1_0
    lfw_paths[3] = pair_1_1
    ...
    and the label:
    issame_list[0] = 1  (means that pair_0 is the sample person)
    issame_list[1] = 0
    ...
    '''

    i = 0
    for path in lfw_paths:
        with open(path, 'rb') as fin:
            
            _bin = fin.read()
            lfw_bins.append(_bin)
            #img = mx.image.imdecode(_bin)
            #img = nd.transpose(img, axes=(2, 0, 1))
            #lfw_data[i][:] = img
            i+=1
            if i%1000==0:
                print('loading dataset', i)

    # with open(args.output, 'wb') as f:
    # pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
