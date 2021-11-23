#This code is modified from https://github.com/JiaFong/cvprw2020-cross-domain-few-shot-learning-challenge/methods/protonet.py
import methods.backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from methods.meta_template import MetaTemplate
from tensorboardX import SummaryWriter
import utils
from methods.loss import PrototypeTripletLoss
import ipdb


class MyMethod(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, tf_path=None, ptl=False):
        super(MyMethod, self).__init__( model_func,  n_way, n_support, tf_path=tf_path)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.n_way = n_way
        
        self.ptl = ptl
        if self.ptl:
            self.ptloss_fn = PrototypeTripletLoss(n_way, n_support)

        self.mean_fc = nn.Linear(512, 128)
        self.std_fc = nn.Linear(512, 128)
        

    def correct(self, x):       
        loss = self.set_forward_loss(x)
        scores  = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )
        if len(scores) > 1:
            scores = scores[0]


        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
    
        return float(top1_correct), len(y_query), loss.item()*len(y_query)



    def parse_feature(self,x,is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        
        _mean = self.mean_fc(z_all)
        _std = self.std_fc(z_all)

        return z_support, z_query, _mean, _std



    def set_forward(self,x,is_feature = False):
        
        z_support, z_query, mu, logvar  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        if self.ptl:
            return scores, z_support, z_proto, mu, logvar
        else:
            return scores


    def set_forward_loss(self, x):
        
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        if self.ptl:
            scores, z_support, z_proto, mu, logvar = self.set_forward(x)
            
            kl_div = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean()
            return self.loss_fn(scores, y_query) + self.ptloss_fn(z_support, z_proto) + 0.1 * kl_div
        else:
            scores = self.set_forward(x)
            return self.loss_fn(scores, y_query)


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)