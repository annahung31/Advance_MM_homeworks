import numpy as np
import os
import random
import torch
from data.datamgr import SetDataManager, SimpleDataManager
from options import parse_args, get_resume_file, load_warmup_state
from methods.LFTNet import LFTNet
from datasets import cifar100_few_shot
import ipdb

def cycle(iterable):
  while True:
    for x in iterable:
      yield x



# training iterations
def train(train_loaders, base_set, train_iters, val_loaders, model, start_epoch, stop_epoch, params):

  # for validation
  max_acc = 0
  total_it = 0

  # training
  for epoch in range(start_epoch,stop_epoch):
    

    # randomly split seen domains to pseudo-seen and pseudo-unseen domains


    if base_set[0] == base_set[1]:
      pu_loader = ps_loader = train_loaders[0]
      aux_iter = train_iters[0]
      

    else:
      random_set = random.sample([0, 1], k=2)
      ps_set = random_set[0]
      pu_set = random_set[1]
      print('seen: {}, unseen: {}'.format(ps_set, pu_set))
      ps_loader = train_loaders[ps_set]
      pu_loader = train_loaders[pu_set]
      

    aux_iter =  train_iters[0]
    val_loader = val_loaders[0]


    # train loop
    model.train()
    total_it = model.trainall_loop(epoch, ps_loader, pu_loader, aux_iter, total_it)

    # validate
    model.eval()
    with torch.no_grad():
      acc = model.test_loop(val_loader)

    # save
    if acc > max_acc:
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      model.save(outfile, epoch)
    else:
      print('GG!! best accuracy {:f}'.format(max_acc))
    if ((epoch + 1) % params.save_freq==0) or (epoch == stop_epoch - 1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch + 1))
      model.save(outfile, epoch)

  return


# --- main function ---
if __name__=='__main__':

  # set numpy random seed
  np.random.seed(10)

  # parse argument
  params = parse_args('train')
  print('--- LFTNet training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  if params.dataset == 'multi':
    #print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
    datasets = ['miniImagenet', 'cifar100']
    
    #datasets.remove(params.testset)
  
  else:
    datasets = [params.dataset, params.dataset]
  
  val_file = os.path.join(params.data_dir, 'miniImagenet', 'LFT', 'val.json')
  
  # model
  print('\n--- build LFTNet model ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224

  n_query = max(1, int(16* params.test_n_way/params.train_n_way))
  train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot)

  train_loaders = []
  train_iters = []
  #train loader: miniImageNet
  base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
  aux_datamgr             = SimpleDataManager(image_size, batch_size=16)
  aux_iter              = iter(cycle(aux_datamgr.get_data_loader(os.path.join(params.data_dir, 'miniImagenet', 'LFT', 'base.json'), aug=params.train_aug)))
  
  train_loaders.append(base_datamgr.get_data_loader(os.path.join(params.data_dir, 'miniImagenet', 'LFT', 'base.json'), aug=params.train_aug))
  train_iters.append(aux_iter)



  #train loader: cifer100
  if params.dataset == 'multi':
    cifar100_datamgr        = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
    cifar100_aux_datamgr    = cifar100_few_shot.SimpleDataManager(image_size, batch_size=16)
    cifar100_aux_iter       = iter(cycle(cifar100_aux_datamgr.get_data_loader(aug=params.train_aug)))
    
    train_loaders.append(cifar100_datamgr.get_data_loader(aug = params.train_aug))
    train_iters.append(cifar100_aux_iter)


  val_loaders = []
  #test loader: miniImageNet
  test_few_shot_params    = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)
  val_loaders.append(val_loader)

  # #test loader: cifer100
  # if params.dataset == 'multi':
  #   cifar100_val_datamgr    = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
  #   val_loaders.append(cifar100_val_datamgr.get_data_loader(aug = False))



  model = LFTNet(params, tf_path=None)
  model.cuda()

  # resume training
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      start_epoch = model.resume(resume_file)
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
      raise ValueError('No resume file')
  # load pre-trained feature encoder
  else:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide pre-trained feature-encoder file using --warmup option!')
    model.model.feature.load_state_dict(load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method), strict=False)

  # training
  print('\n--- start the training ---')
  train(train_loaders, datasets, train_iters, val_loaders, model, start_epoch, stop_epoch, params)
