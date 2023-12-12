#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:42:02 2023

@author: kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
# from args import args #, device

device = 'cuda' if torch.cuda.is_available() else 'cpu'


## Import local modules (Prof.JX-W's python code)
# RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
# RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
# sys.path.append(RWOF_dir)
# sys.path.append(RWOF_dir_1)

# from Lang_sampler import Lang_sampler

class LangD(object):

    def __init__(self, bayes_nn, n_samples, lr, delta, train_loader1):
        self.bayes_nn = bayes_nn
        self.train_loader1 = train_loader1
        self.n_samples = n_samples
        self.lr = lr 
        self.delta = delta
        # self.optimizers1, self.optimizers2 = self._optimizers_schedulers()
        # self.ini_stats = torch.load('AdamOpt_results_2D_PDE_stats_10trials.pt')
        self.optimizers = self._optimizers_schedulers()
        # self.priors = []
        # for i in range(self.n_samples):
        #     self.priors.append( torch.load('AdamOpt_results_2D_PDE_results_10trials.pt')[str(i+1)] )

    def _optimizers_schedulers(self):
        optimizers = []
        # optimizers1 = []
        # optimizers2 = []
        for i in range(self.n_samples):
            lr = self.lr
            delta = self.delta
            parameters = self.bayes_nn.nnets[i].features.parameters()
            # optimizer_i = torch.optim.LBFGS(parameters, lr=lr)
            optimizer_i = torch.optim.Adam(parameters, lr=lr) # Lang_sampler(self.bayes_nn.nnets[i], parameters, lr=lr, delta=delta) 
            optimizers.append(optimizer_i)
            # optimizer1_i = torch.optim.Adam(parameters, lr=lr)
            # parameters = self.bayes_nn.nnets[i].features.parameters()
            # optimizer2_i = Lang_sampler(self.bayes_nn.nnets[i], parameters, lr=lr, delta=delta)
            # optimizers1.append(optimizer1_i)
            # optimizers2.append(optimizer2_i)
        return optimizers
        # return optimizers1, optimizers2
    
    # def cal_prior(self, i):

    #     params = self.bayes_nn.nnets[i].features.parameters()
    #     # priors = self.priors[i]
    #     ini_stats = self.ini_stats 
        
    #     # priors = self.priors
    #     norm = 0
    #     # lamb = 0.01
    #     for (ind,p) in enumerate(params):
    #         mean = ini_stats[0,ind]
    #         std = ini_stats[1,ind]
    #         mean_t = torch.full(p.shape, mean)
    #         norm += - 1/(2*std**2) *  torch.Tensor.norm( (p.data-mean_t) ,2)**2
    #         # norm += - lamb/2 *  torch.Tensor.norm(p.data,2)**2
    #         # norm += - 1/(2*lamb**2) *  torch.Tensor.norm( (p.data-priors[ind]) ,2)**2
    #         # norm += - 1/(2*lamb**2) *  torch.Tensor.norm( p.data ,2)**2
    #     return norm
    
    def cal_likelihood(self, i, t, x, y, z, out, dim):
        t = torch.FloatTensor(t).to(device)
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        z = torch.FloatTensor(z).to(device)
        out = torch.FloatTensor(out).to(device)
        out_d = out.detach()
        
        inp = torch.cat((t,x,y,z), 1)
        invRe = self.bayes_nn.nnets[i].invRe
        out_hat = self.bayes_nn.nnets[i].forward(inp)
        
        P_hat = torch.reshape(out_hat[:,0], [out_hat.shape[0],1])
        u_hat = torch.reshape(out_hat[:,1], [out_hat.shape[0],1])
        v_hat = torch.reshape(out_hat[:,2], [out_hat.shape[0],1])
        w_hat = torch.reshape(out_hat[:,3], [out_hat.shape[0],1])
        
        P_hat_x = torch.autograd.grad(P_hat, x, grad_outputs=torch.ones_like(P_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        P_hat_y = torch.autograd.grad(P_hat, y, grad_outputs=torch.ones_like(P_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        P_hat_z = torch.autograd.grad(P_hat, z, grad_outputs=torch.ones_like(P_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        
        u_hat_t = torch.autograd.grad(u_hat, t, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_y = torch.autograd.grad(u_hat, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_yy = torch.autograd.grad(u_hat_y, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_z = torch.autograd.grad(u_hat, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        u_hat_zz = torch.autograd.grad(u_hat_z, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        
        v_hat_t = torch.autograd.grad(v_hat, t, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_x = torch.autograd.grad(v_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_xx = torch.autograd.grad(v_hat_x, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_y = torch.autograd.grad(v_hat, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_yy = torch.autograd.grad(v_hat_y, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_z = torch.autograd.grad(v_hat, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        v_hat_zz = torch.autograd.grad(v_hat_z, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        
        w_hat_t = torch.autograd.grad(w_hat, t, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_x = torch.autograd.grad(w_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_xx = torch.autograd.grad(w_hat_x, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_y = torch.autograd.grad(w_hat, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_yy = torch.autograd.grad(w_hat_y, y, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_z = torch.autograd.grad(w_hat, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        w_hat_zz = torch.autograd.grad(w_hat_z, z, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        
        L1 = u_hat_t + u_hat*u_hat_x + v_hat*u_hat_y + w_hat*u_hat_z + P_hat_x - (u_hat_xx + u_hat_yy + u_hat_zz) * invRe
        L2 = v_hat_t + u_hat*v_hat_x + v_hat*v_hat_y + w_hat*v_hat_z + P_hat_y - (v_hat_xx + v_hat_yy + v_hat_zz) * invRe
        L3 = w_hat_t + u_hat*w_hat_x + v_hat*w_hat_y + w_hat*w_hat_z + P_hat_z - (w_hat_xx + w_hat_yy + w_hat_zz) * invRe
        L4 = u_hat_x + v_hat_y + w_hat_z
        
        noise = 0.01
        pi = np.pi
        log_like_1 = - 0.5*( ( torch.reshape(out_d[:,0], u_hat.shape) - u_hat)/noise )**2 - 0.5*np.log(2*pi*noise**2)
        log_like_2 = - 0.5*( ( torch.reshape(out_d[:,1], v_hat.shape) - v_hat)/noise )**2 - 0.5*np.log(2*pi*noise**2)
        log_like_3 = - 0.5*( ( torch.reshape(out_d[:,2], w_hat.shape) - w_hat)/noise )**2 - 0.5*np.log(2*pi*noise**2)
        log_like_4 = - 0.5*( (L1)/noise )**2 - 0.5*( (L2)/noise )**2 - 0.5*( (L3)/noise )**2 - 0.5*( (L4)/noise )**2 
        log_like = log_like_1 + log_like_2 + log_like_3 +log_like_4
        return log_like
    
    
    def compute2(self, i, x):
        x = torch.FloatTensor(x).to(device)
        u_hat = self.bayes_nn[i].forward(x)
        u_hat = u_hat.detach()
        return u_hat
    
    
    def train(self, epoch, epochs, dim):
        
        self.bayes_nn.train() # prepares the model for training (but does NOT actually train the model)
        
        log_like_f = 0
        log_prior = 0
        log_post = 0
        Log_post = []
        Log_like_f = []
        Log_prior = []
        
        #self.bayes_nn.zero_grad() # zero the parameter gradients
        Log_post = torch.zeros([self.n_samples])
        Log_like_f = torch.zeros([self.n_samples])
        Log_like_b = torch.zeros([self.n_samples])
        Log_prior = torch.zeros([self.n_samples])
        for i in range(self.n_samples):
            
            self.bayes_nn.nnets[i].zero_grad()  # zero the parameter gradients
            
            # log_prior = self.cal_prior(i)
            
            log_like_f = torch.zeros([1])
            for batch_id, (inp,out) in enumerate(self.train_loader1):
                t = torch.reshape(inp[:,0], [inp[:,0].shape[0],1])
                x = torch.reshape(inp[:,1], [inp[:,1].shape[0],1])
                y = torch.reshape(inp[:,2], [inp[:,2].shape[0],1])
                z = torch.reshape(inp[:,3], [inp[:,3].shape[0],1])
                t.requires_grad = True
                x.requires_grad = True
                y.requires_grad = True
                z.requires_grad = True
                out = torch.reshape(out, [out.shape[0], out.shape[1]])
                
                log_like_f_all = self.cal_likelihood(i, t, x, y, z, out, dim)
                log_like_f += log_like_f_all.sum() # sum of all log likelihoods

            log_post = - log_like_f 
            # log_post = log_like_f + log_prior
            self.optimizers[i].zero_grad()
            
            log_post.backward()
            log_post_d = log_post.detach()
            log_like_f_d = log_like_f.detach()
            # log_prior_d = log_prior.detach()
            # Log_post.append(log_post_d)
            # Log_like_f.append(log_like_f_d)
            # Log_like_b.append(log_like_b_d)
            # Log_prior.append(log_prior_d)
            Log_post[i] = log_post_d
            Log_like_f[i] = log_like_f_d
            #Log_prior[i] = log_prior_d
            
            # options = {'closure': self.cal_likelihood, 'current_loss': log_post}
            self.optimizers[i].step()
            # self.optimizers[i].step(epoch, i, 'Lang')
            
        mean_log_post = np.average(Log_post)
        mean_log_like_f = np.average(Log_like_f)
        mean_log_like_b = np.average(Log_like_b)
        mean_log_prior = np.average(Log_prior)
        return mean_log_post, mean_log_like_f, mean_log_like_b, mean_log_prior
    
    
    def compute(self, test_loader):
        u_hat_all = []
        # f_hat_all = []
        for i in range(self.n_samples):
            
            for batch_id, (inp,out) in enumerate(test_loader):
                u_hat = self.bayes_nn.nnets[i].forward(inp)
            
            u_hat_all.append(u_hat.detach())
            # f_hat_all.append(f_hat.detach())
        return u_hat_all #, f_hat_all
    
        
    
        
        
        
        