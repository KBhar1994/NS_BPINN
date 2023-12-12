#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:21:04 2023

@author: kevin
"""

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

class Lang_sampler(torch.optim.Optimizer):
      
    def __init__(self, bayes_nn, params, lr, delta):
        super(Lang_sampler, self).__init__(params, defaults={'lr': lr})
        self.bayes_nn = bayes_nn
        self.params = params
        self.lr = lr
        self.delta = delta
    
    def step(self, epoch):
        params = self.bayes_nn.parameters() # all unknown parameters
        co = 0
        gamma = 1 / 230.0
        alpha = self.lr / (gamma*epoch + 1) ** self.delta
        
        for p in params:
            co += 1
            grad = p.grad
            v = torch.normal(0, torch.sqrt( torch.tensor(2*alpha) ), size=p.data.size() )
            p.data += - alpha*grad + v
            
            
            
            
            
            
            
            