import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import ipdb
import h5py

from data.datamgr import SimpleDataManager, SetDataManager
import data.feature_loader as feat_loader
from methods.protonet import ProtoNet
from methods.myMethod import MyMethod
from methods.LFTNet import LFTNet
from methods.loss import *

from methods.backbone import model_dict
from options import parse_args, get_resume_file, get_best_file, get_assigned_file, load_warmup_state

from utils import *

from pseudo_query_generator import PseudoQeuryGenerator
from datasets import CropDisease_few_shot, EuroSAT_few_shot, ISIC_few_shot


params = parse_args('test')


# extract and save image features
def save_features(model, batch, featurefile):

    x, y = batch
    x = x.cuda()
    x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
    y = y.reshape(y.shape[0]*y.shape[1]) 
    max_count = x.shape[0]

    f = h5py.File(featurefile, 'w')
    
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0

    feats = model(x)
    if all_feats is None:
        all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
    
    # ipdb.set_trace()
    all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
    all_labels[count:count+feats.size(0)] = y.cpu().numpy()
    count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()




def get_feature(metric_model, batch, checkpoint_dir):
    model = model_dict[params.model]()
    model = model.cuda()

    state = metric_model.state_dict()
    state_keys = list(state.keys())

    for i, key in enumerate(state_keys):
        if "feature." in key and not 'gamma' in key and not 'beta' in key:
            newkey = key.replace("feature.","")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()


    # save feature file
    print('  extract and save features...')
    if params.save_epoch != -1:
        featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"),  "test_" + str(params.save_epoch)+ ".hdf5")
    else:
        featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"),  "test.hdf5")
    dirname = os.path.dirname(featurefile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, batch, featurefile)

    return featurefile




def meta_test(novel_loader, seen_loader=None, n_query = 15, task='fsl', finetune=True, n_pseudo=100, n_way = 5, n_support = 5): 
    correct = 0
    count = 0
    # saver_agent.global_step = 0

    iter_num = len(novel_loader) 

    acc_all = []
    for ti, (x, y) in enumerate(novel_loader):
        
        ###############################################################################################
        # load pretrained model on miniImageNet
        if params.method == 'protonet':
            pretrained_model = ProtoNet(model_dict[params.model], n_way = n_way, n_support = n_support)
            
        elif params.method == 'myMethod':
            pretrained_model = MyMethod(model_dict[params.model], n_way = n_way, n_support = n_support, ptl=True)

        else:
            raise ValueError 


        #change the folder name if the baseline model is trained using KLD loss
        if params.kl > 0:
            task_path = 'single_kl' if task in ["fsl", "cdfsl-single"] else 'multi_kl'
            checkpoint_dir = '%s/checkpoints/%s' %(params.save_dir, task_path)

        else:
            task_path = 'single' if task in ["fsl", "cdfsl-single"] else 'multi'
            checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(params.save_dir, task_path, params.model, params.method)
            if params.train_aug:
                checkpoint_dir += '_aug'
            checkpoint_dir += '_5way_5shot'        
        
        
        assert os.path.exists(checkpoint_dir)
        


        params.save_iter = -1
        if params.save_epoch != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            try:
                pretrained_model.load_state_dict(tmp['state'])
            except RuntimeError:
                print('warning! RuntimeError when load_state_dict()!')
                pretrained_model.load_state_dict(tmp['state'], strict=False)
            except KeyError:
                for k in tmp['model_state']:   ##### revise latter
                    if 'running' in k:
                        tmp['model_state'][k] = tmp['model_state'][k].squeeze()
                pretrained_model.load_state_dict(tmp['model_state'], strict=False)
            except:
                raise

        pretrained_model.cuda() 


        
        # if params.data_parallel > 0:
        #     pretrained_model = nn.DataParallel(pretrained_model, device_ids=[0,1])
        ###############################################################################################
        # split data into support set and query set
        n_query = x.size(1) - n_support
        
        x = x.cuda()
        x_var = Variable(x)

        support_size = n_way * n_support 
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda()    # (25,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # query set
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])  # support set
 
        
        if finetune:
            ##TODO: update or not update ft? that is, use things in trainall_loop or train_loop?
            ###############################################################################################
            # Finetune components initialization 
            pseudo_q_genrator  = PseudoQeuryGenerator(n_way, n_support,  n_pseudo)
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()))

            ###############################################################################################
            # finetune process 
            finetune_epoch = 100
        
            fine_tune_n_query = n_pseudo // n_way
            pretrained_model.n_query = fine_tune_n_query
            pretrained_model.train()

            z_support = x_a_i.view(n_way, n_support, *x_a_i.size()[1:])
                
            for epoch in range(finetune_epoch):
                
                delta_opt.zero_grad()

                # generate pseudo query images
                psedo_query_set, _ = pseudo_q_genrator.generate(x_a_i)
                psedo_query_set = psedo_query_set.cuda().view(n_way, fine_tune_n_query,  *x_a_i.size()[1:])

                x = torch.cat((z_support, psedo_query_set), dim=1)
 
                
                loss = pretrained_model.set_forward_loss(x)
                loss.backward()
                delta_opt.step()
                
                #empty cache to avoid OOM
                torch.cuda.empty_cache()

        ###############################################################################################
        # inference 
        print('Inference...')
        batch = (x_var, y)

        #get feature using feature transform layers
        featurefile = get_feature(pretrained_model, batch, checkpoint_dir)
        cl_data_file = feat_loader.init_loader(featurefile)
        class_list = cl_data_file.keys()
        #take class features
        z_all = []
        for cl in class_list:
            
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
        z_all = torch.from_numpy(np.array(z_all) )

        pretrained_model.eval()        
        pretrained_model.n_query = n_query
        with torch.no_grad():
            
            scores = pretrained_model.set_forward(z_all.cuda(), is_feature = True)

            if type(scores) is tuple:   #if apply PTLoss during meta-training
                scores = scores[0]

        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)

        acc_all.append((correct_this/ count_this *100))        
        print("Task %d : %4.2f%%  Now avg: %4.2f%%" %(ti, correct_this/ count_this *100, np.mean(acc_all) ))
        # saver_agent.add_summary('Task %d, current Avg'%(ti), np.mean(acc_all))
        ###############################################################################################
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    # saver_agent.add_summary('Test Acc', '%4.2f%% +- %4.2f%%'%(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    
    task = params.task

    # path_exp = os.path.join(configs.finetune_log_dir, params.exp_name)    
    #init the saver
    # saver_agent = saver.Saver(path_exp)
    # saver_agent.add_summary_msg(
    #     ' > task: {}'.format(task))
    # saver_agent.add_summary_msg(
    #     ' > Method: {}'.format(params.method))
    
    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way))
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   

    # number of pseudo images
    n_pseudo = 100

    ##################################################################
    dataset_names = ["miniImageNet"]
    print ("Loading mini-ImageNet")
    datamgr        = SetDataManager(image_size, n_eposide = iter_num, n_query = 15,  **few_shot_params)
    miniImg_loader   = datamgr.get_data_loader(os.path.join(params.data_dir, 'miniImagenet', 'LFT', 'novel.json'), aug =False)

    novel_loaders = []
    if task == 'fsl':
        
        finetune = False

        print('fine-tune: ', finetune)
        if finetune:
            print("n_pseudo: ", n_pseudo)

        meta_test(miniImg_loader, n_query = 15, task=task, finetune=finetune, n_pseudo=n_pseudo, **few_shot_params)


    else:
        finetune = params.finetune
        dataset_names = ["CropDisease", "EuroSAT", "ISIC"]
        
        print ("Loading CropDisease")
        datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        CropDisease_loader  =  datamgr.get_data_loader(aug =False)
        novel_loaders.append(CropDisease_loader)
        
                    

        print ("Loading EuroSAT")
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        EuroSAT_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(EuroSAT_loader)

        print ("Loading ISIC")
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        ISIC_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(ISIC_loader)
        


        print('fine-tune: ', finetune)
        if finetune:
            print("n_pseudo: ", n_pseudo)

        #########################################################################
        # meta-test loop
        for idx, novel_loader in enumerate(novel_loaders):
            print (dataset_names[idx])
            meta_test(novel_loader, seen_loader=miniImg_loader, n_query = 15, task=task, finetune=finetune, n_pseudo=n_pseudo, **few_shot_params)
