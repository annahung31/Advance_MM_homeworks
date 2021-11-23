

import glob
import torch
from utils.utils_callbacks import CallBackTest
from backbones import iresnet50
import pandas as pd
from data_preparing import prepare_test
import ipdb

def gen_test_pair_list(label_path, pair_path):

    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    labels = [x.rstrip() for x in labels]

    ## prepare for test_anna
    datas = []
    fname_prefix = 'test_pair_'
    for i in range(len(labels)):
        f1 = fname_prefix + str(i) + '_' + str(1) + '.jpg'
        f2 = fname_prefix + str(i) + '_' + str(2) + '.jpg'
        label = labels[i]
        datas.append(f1 + ' ' + f2 + ' ' + label + '\n')

    
    with open(pair_path, 'w') as f:
        f.writelines(datas) 
    

def verificate(label_path, pair_path, img_path, test_model_path, mode='closed'):
    gen_test_pair_list(label_path, pair_path)

    model = iresnet50()
    model.load_state_dict(torch.load(test_model_path))
    # model.to(torch.device("cuda"))
    print('testing model:', test_model_path)
    testing = CallBackTest(rank=0, pair_path=pair_path, img_path=img_path)
    acc, predict = testing(model)

    name = test_model_path.split('/')[-2]
    with open(mode + '_' + name + '_predict.txt', 'w') as f:
        for p in range(len(predict)):
            f.write(str(predict[p].astype(int)) + '\n')

    del model
    return acc





if __name__ == "__main__":

    img_path = 'data/test/closed_set/test_pairs_align/test_pairs'
    pair_path = 'closed_test_pairs.txt'
    label_path = 'data/test/closed_set/labels.txt'
    single_model = False
    if single_model:
        test_model_path = 'models/0414-1916/backbone_2232.pth'
        acc = verificate(label_path, pair_path, img_path, test_model_path, mode='closed')

    else:
        model_id = []
        accs = []
        max_acc = [-1, '']

        models = glob.glob('models/0415-1222/backbone_*.pth')
        for test_model_path in models:
            acc = verificate(label_path, pair_path, img_path, test_model_path, mode='closed')
            model_id.append(int(test_model_path.split('_')[-1].split('.')[0]))
            accs.append(acc)
            
            if max_acc[0] < acc:
                max_acc[0] = acc
                max_acc[1] = test_model_path
            


        df = pd.DataFrame({'id': model_id, 'acc': accs})
        df = df.sort_values(by ='id')
        df.to_csv('models/0415-1222/closed_accuracy.csv')    

        print('Best model: {}, acc: {}'.format(max_acc[1], max_acc[0]))
