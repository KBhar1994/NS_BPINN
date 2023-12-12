#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:13:04 2023

@author: kevin
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
import subprocess # Call the command line
from subprocess import call
import pdb
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau

## Import local modules (Prof.JX-W's python code)
#RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
#RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
#sys.path.append(RWOF_dir)
#sys.path.append(RWOF_dir_1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        

# import the modules you need
#import foamFileOperation as foamOp
class Swish(nn.Module): # defines the Swish activation function 
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
            
            
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_output):
        super(Net, self).__init__()
        
        # init_params = torch.load('AdamOpt_results_2D_PDE_results.pt')
        self.invRe = torch.nn.Parameter( torch.abs(torch.normal(mean=torch.tensor(1e-5), std=torch.tensor(1e-5))) )
        self.invRe.retain_grad = True
        
        self.features = nn.Sequential() # creates sequence of NN layers called 'features' (add_module() adds a specific layer to the sequence)
        self.features.add_module('hidden', torch.nn.Linear(n_feature, n_hidden_1))
        self.features.hidden.retain_grad = True
        # self.features.hidden.weight.data = init_params[0].data
        # self.features.hidden.bias.data = init_params[1].data
        # self.features.add_module('active1', Swish())
        self.features.add_module('active1', nn.Tanh())
        
        self.features.add_module('hidden2', torch.nn.Linear(n_hidden_1, n_hidden_2))
        self.features.hidden2.retain_grad = True
        # self.features.hidden2.weight.data = init_params[2].data
        # self.features.hidden2.bias.data = init_params[3].data
        # self.features.add_module('active2', Swish())
        self.features.add_module('active2', nn.Tanh())
        
        self.features.add_module('predict', torch.nn.Linear(n_hidden_2,  n_output))
        self.features.predict.retain_grad = True
        # self.features.predict.weight.data = init_params[4].data
        # self.features.predict.bias.data = init_params[5].data
        # self.features.add_module('active3', Swish())
        self.features.apply(init_weights)
        
    def forward(self, x):
        # print('FCN forward() function')
        return self.features(x) # makes a forward pass through 'features' with input 'x'
    
    def reset_parameters(self, verbose=False): # NOT SURE what this does
        #TODO: where did you define module?
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
        if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):


                module.reset_parameters()
            if verbose:
                print("Reset parameters in {}".format(module))
                
                
                
                
                
                
                
                
                
                