#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:28:03 2023

@author: kevin
"""


import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sp
import copy
import math
import netCDF4 as nc
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
# from time import time
import time
import sys
import os
import gc
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
import NN_2DNS
from BayesNN2 import BayesNN2
from LangD_2DNS import LangD


# #%% Prepare the training data

# def PrepareData(Tf = 20, N = 100, random=True):
    
#     # load N data points from the first Tf seconds
#     ALT = 5
#     xdim = 40
#     ydim = 40
#     xsize = 0.4 #0.4 km
#     ysize = 0.4 #0.4 km
#     #total_dim = xdim * ydim
    
#     xt = np.zeros([Tf*N,1])
#     yt = np.zeros([Tf*N,1])
#     tt = np.zeros([Tf*N,1])
#     ut = np.zeros([Tf*N,1])
#     vt = np.zeros([Tf*N,1])
#     pt = np.zeros([Tf*N,1])
    
#     Fstart = 3
    
#     for i in range(Tf):
#         ind = i + Fstart
#         if ind  >= 10:
#             F = 'cm1out_0000' + str(ind) + '.nc'
#         # elif i >= 10 and i < 100:
#         #     F = 'wind_0' + str(i) + '.txt'
#         elif ind < 10: 
#             F = 'cm1out_00000' + str(ind) + '.nc'
#         F = os.path.join(os.getcwd(), 'ncfiles',F)
#         ds = nc.Dataset(F)
#         if i + Fstart == 3:
#             # start of the time 
#             tstart = ds['time'][:]
#             ti = 0
#         else:
#             ti = ds['time'][:] - tstart
        
#         tt[i*N:(i+1)*N,:] = ti/30 * np.ones([N,1])
        
        
#         U = ds['uinterp'][:]
#         V = ds['vinterp'][:]
#         P = ds['prs'][:]
#         xh = ds['xh'][:].reshape([xdim,1])
#         yh = ds['yh'][:].reshape([ydim,1])

        
#         S = int(np.sqrt(N))
        
#         xidx = np.kron(np.random.choice(xdim, S, replace=False).reshape([S,1]), np.ones([S,1]))  
#         xs = xh[xidx.astype(int)]/xsize
        
#         yidx = np.kron(np.ones([S,1]), np.random.choice(ydim, S, replace=False).reshape([S,1]))
#         ys = yh[yidx.astype(int)]/ysize

        
#         ut[i*N:(i+1)*N,:] = U[:, ALT, yidx.astype(int),xidx.astype(int)].reshape([N,1])
#         vt[i*N:(i+1)*N,:] = V[:,ALT, yidx.astype(int),xidx.astype(int)].reshape([N,1])
#         pt[i*N:(i+1)*N,:] = P[:,ALT, yidx.astype(int),xidx.astype(int)].reshape([N,1])/100000.0

#         xt[i*N:(i+1)*N,:] = xs.reshape([N,1])
#         yt[i*N:(i+1)*N,:] = ys.reshape([N,1])
        
#     return (xt,yt,tt,ut,vt,pt)

#%% Load the training data and add noise

Data = sp.loadmat('2DNS_data.mat')
x_train = Data['x']
y_train = Data['y']
t_train = Data['t']
u_train = Data['u']
v_train = Data['v']
P_train = Data['P']
ts = Data['ts']
sp = Data['sp']

# Add noise
std = 0.1
noise_d = np.random.normal(0, std, size = u_train.shape)
u_train += noise_d
noise_d = np.random.normal(0, std, size = v_train.shape)
v_train += noise_d
noise_d = np.random.normal(0, std, size = P_train.shape)
P_train += noise_d

#%% Training  the model

Training = True;

n_samples = 5
n_feature = 3
n_hidden = 20 
n_output = 3
epochs = 5000  
lr = 1e-2 
delta = 0.01
noise = 1e-6

# T_train = 20 # number of snapshots
# num = 1600 # number of spatial points in each snapshot
# dim = T_train*num # total number trianing data points
# x_train, y_train, t_train, u_train, v_train, P_train = PrepareData(T_train, num, random=True)

dim = ts * sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
denseFCNN = NN_2DNS.Net(n_feature, n_hidden, n_output)
bayes_nn = BayesNN2(denseFCNN, n_samples=n_samples, noise=noise).to(device)

inp = np.concatenate((t_train,x_train,y_train), axis=1)
out = np.concatenate((u_train,v_train, P_train), axis=1)
# out = np.concatenate((u_train,v_train), axis=1)

data = torch.utils.data.TensorDataset(torch.FloatTensor(inp), torch.FloatTensor(out))
train_loader1 = torch.utils.data.DataLoader(data, batch_size=int(dim/300), shuffle=True)

langd = LangD(bayes_nn, n_samples, lr, delta, train_loader1, std)
print('LangD initialized')

loss = []
log_post = []
log_like_f = []
log_like_b = []
log_prior = []

if Training == True:
    print('Training starting...')
    for epoch in range(epochs):
        mean_log_post, mean_log_like_f, mean_log_like_b, mean_log_prior = langd.train(epoch, epochs, dim)
        log_post.append(mean_log_post)
        log_like_f.append(mean_log_like_f)
        log_like_b.append(mean_log_like_b)
        log_prior.append(mean_log_prior)
            
        if epoch % 100 == 0:
            # print('Epochs complete:', epoch, '/', epochs)
            loss = np.array(log_post)
            np.save('log_like_loss_Lang_2DNS.npy', loss)
    print('Training finished...')

    P = []
    for i in range(n_samples):
        params = []
        tparams = langd.bayes_nn.nnets[i].parameters()
        for p in tparams:
            params.append(p)
        P.append(params)
    
    trained_model = NN_2DNS.Net(n_feature, n_hidden, n_output)
    tparams = trained_model.parameters()
    co = 0
    for p in tparams:
        p.data = torch.zeros(p.shape)
        for i in range(n_samples):
            p.data += P[i][co] / n_samples
        co += 1
        
    # torch.save(langd.bayes_nn.nnets[0], "trained_model_Lang_2DNS.pth")
    torch.save(langd.bayes_nn.nnets, "all_trained_models_Lang_2DNS.pth")
    torch.save(trained_model, "avg_trained_model_Lang_2DNS.pth")










