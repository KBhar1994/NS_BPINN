# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:02:46 2023

@author: Kevin Bhar
"""


# scitific cal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
from time import time
import sys
import os
import gc
import pdb
import subprocess # Call the command line
from subprocess import call
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
# local import
# from NN_struc import Net
# from args import args #, device

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class BayesNN2(nn.Module):
    """Define Bayesian network
    """
    def __init__(self, model, n_samples, noise):
        super(BayesNN2, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(
                torch.typename(model)))
        self.n_samples = n_samples # number of particles (# of perturbed NN)
        
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(model)
            
            # # Manual initialization to MAP estimate
            # # P = torch.load('AdamOpt_results_2D_PDE_results_10trials.pt')[str(i+1)] 
            # P = torch.load('AdamOpt_results_1D_PDE_2.pt')
            # new_instance.features.hidden.weight.data = P[0]
            # new_instance.features.hidden.bias.data = P[1]
            # new_instance.features.hidden2.weight.data = P[2]
            # new_instance.features.hidden2.bias.data = P[3]
            # new_instance.features.predict.weight.data = P[4]
            # new_instance.features.predict.bias.data = P[5]
            
            # # Manual initialization by drawing samples based on optimized values 
            # ini_stats = torch.load('AdamOpt_results_2D_PDE_stats_10trials.pt')
            # size = new_instance.features.hidden.weight.shape
            # new_instance.features.hidden.weight.data = torch.normal(ini_stats[0,0], ini_stats[1,0], size)
            # size = new_instance.features.hidden.bias.shape
            # new_instance.features.hidden.bias.data = torch.normal(ini_stats[0,1], ini_stats[1,1], size)
            
            # size = new_instance.features.hidden2.weight.shape
            # new_instance.features.hidden2.weight.data = torch.normal(ini_stats[0,2], ini_stats[1,2], size)
            # size = new_instance.features.hidden2.bias.shape
            # new_instance.features.hidden2.bias.data = torch.normal(ini_stats[0,3], ini_stats[1,3], size)
            
            # size = new_instance.features.predict.weight.shape
            # new_instance.features.predict.weight.data = torch.normal(ini_stats[0,4], ini_stats[1,4], size)
            # size = new_instance.features.predict.bias.shape
            # new_instance.features.predict.bias.data = torch.normal(ini_stats[0,5], ini_stats[1,5], size)
            
            # Random initialization
            def init_normal(m): # Initialization (Separate initialization for all trials)
                if type(m) == nn.Linear:
                    nn.init.kaiming_normal_(m.weight)
            print(type(new_instance))
            new_instance.apply(init_normal) # applies the function in the argument to each component of the object before dot operator
            print('Reset parameters in model instance {}'.format(i))
            
            # # Random initialization (Xavier distribution)
            # def init_weights(m):
            #     if isinstance(m, nn.Linear):
            #         torch.nn.init.xavier_uniform(m.weight)
            #         m.bias.data.fill_(0.01)
            # new_instance.apply(init_weights) 
            
            instances.append(new_instance) # creates multiple (number of ensembles) copies of the NN structure
            
        self.nnets = nn.ModuleList(instances) # indexed list of pytorch Modules (for each ensemble)
        
        # self.lam = torch.randn(n_samples)
        # for i in range(n_samples):
        #     self.nnets[i].lam = torch.nn.parameter.Parameter( torch.tensor([self.lam[i].item()]) )
        #     self.nnets[i].lam.retain_grad = True
        
        print('Total number of parameters: {}'.format(self._num_parameters()))
    
    
    
    def _num_parameters(self): # calculates the total number of trainable parameter constants in all the ensembles of the NN
        count = 0
        for name, param in self.named_parameters(): # loop over all the parameters of each NN for all ensembles (weights, biases, log_beta)
            # print(name)
            count += param.numel() # param.numel() gives the number of trainable constants in each parameter
        return count

    def __getitem__(self, idx): # returns the 'idx'-th ensemble of the NN ensembles 
        return self.nnets[idx]

    @property
    
    def forward(self, inputs):

        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(inputs)) # here forward() is the function in FCN.py class Net; computes the NN output for 'inputs' 
        output = torch.stack(output) # concatenates a sequence of tensors (of same size) along a new dimension

        return output
    
    
    # def cal_prior(self, i):
    #     params = self.bayes_nn[i].features.parameters()
    #     norm = 0
    #     lamb = 1
    #     for p in params:
    #         norm += - lamb/2 * torch.Tensor.norm(p.data,2)
    #     return norm
        
    # def cal_likelihood_f(self, i, x, f, ntrain):
    #     x = torch.FloatTensor(x).to(device)
    #     f = torch.FloatTensor(f).to(device)
    #     f_d = f.detach()
    #     # x.requires_grad = True
    #     pi = np.pi
    #     u_hat = self.nnets[i].forward(x)
    #     u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x),create_graph = True, only_inputs=True)[0]
    #     u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    #     f_hat = 0.01 * u_hat_xx
    #     f_hat_d = f_hat.detach()
    #     # likelihood = 1/ np.sqrt(2*pi*0.1**2) * np.exp( -  np.power( (f - f_hat_d), 2 ) / (2*0.1**2) )
    #     # log_like = np.log(likelihood)
    #     log_like = - 0.5*( (f_d - f_hat_d)/0.01 )**2 - 0.5*np.log(2*pi*0.01**2)
    #     return log_like
        
    # def cal_likelihood_b(self, i, x, u, ntrain):
    #     x = torch.FloatTensor(x).to(device)
    #     u = torch.FloatTensor(u).to(device)
    #     u_d = u.detach()
    #     # x.requires_grad = True
    #     pi = np.pi
    #     u_hat = self.nnets[i].forward(x)
    #     u_hat_d = u_hat.detach()
    #     # likelihood = 1/ np.sqrt(2*pi*0.1**2) * np.exp( -  np.power( (u - u_hat_d), 2 ) / (2*0.1**2) )
    #     # log_like = np.log(likelihood)
    #     log_like = - 0.5*( (u_d - u_hat_d)/0.01 )**2 - 0.5*np.log(2*pi*0.01**2)
    #     return log_like
    
    def compute2(self, i, x):
        x = torch.FloatTensor(x).to(device)
        u_hat = self.nnets[i].forward(x)
        u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x),create_graph = True, only_inputs=True)[0]
        # u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        f_hat = 0.01 * u_hat_x
        u_hat = u_hat.detach()
        f_hat = f_hat.detach()
        return u_hat, f_hat
    
    
    