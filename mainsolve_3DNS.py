#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:00:26 2023

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:14:29 2023

@author: Kevin Bhar
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sp
import copy
import math
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
# local import
import NN_3DNS
from BayesNN2 import BayesNN2
from LangD_3DNS import LangD


# 0.01 d^2u/dx^2 = f
# Solution : u(x) = sin^3(6x)
# f(x) =  (108 * 0.01) sin(6*x) ( 2*cos^2(6*x) - sin^2(6*x) )
 

Training = True;

n_samples = 1
n_feature = 4
n_hidden_1 = 50 
n_hidden_2 = 50 
n_output = 4
epochs = 3000  
lr = 1e-3 # (LD:1e-6 (random ini); LD:1e-8 (Adam opt ini); HMC:1e-2; ADAM:1e-4)   
delta = 0.01
noise = 1e-6
# Re = 1e3
method = 'Lang'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
denseFCNN = NN_3DNS.Net(n_feature, n_hidden_1, n_hidden_2, n_output)
bayes_nn = BayesNN2(denseFCNN, n_samples=n_samples, noise=noise).to(device)

ts = 10
xp = 40
yp = 40
zp = 40
dim = ts*xp*yp*zp

Data = sp.loadmat('sbl_coarse_truncated.mat')
u = Data['u'][0:10,:,:,:] 
v = Data['v'][0:10,:,:,:] 
w = Data['w'][0:10,:,:,:] 
x = Data['x'] 
y = Data['y'] 
z = Data['z'] 
t = Data['time'][:,0:10]

x = np.reshape(x, [x.shape[1],1])
y = np.reshape(y, [y.shape[1],1])
z = np.reshape(z, [z.shape[1],1])
t = np.reshape(t, [t.shape[1],1])

co = 0
tt = np.zeros([dim,1])
xt = np.zeros([dim,1])
yt = np.zeros([dim,1])
zt = np.zeros([dim,1])
for i in range(len(t)):
    for j in range(len(z)):
        for k in range(len(y)):
            for l in range(len(x)):
                zt[co,0] = z[j,0]
                yt[co,0] = y[k,0]
                xt[co,0] = x[l,0]
                tt[co,0] = t[i,0]
                co += 1

ut = np.reshape(u, [dim,1])
vt = np.reshape(v, [dim,1])
wt = np.reshape(w, [dim,1])

inp = np.concatenate((tt,xt,yt,zt), axis=1)
out = np.concatenate((ut,vt,wt), axis=1)

data = torch.utils.data.TensorDataset(torch.FloatTensor(inp), torch.FloatTensor(out))
train_loader1 = torch.utils.data.DataLoader(data, batch_size=int(dim/300), shuffle=True)

langd = LangD(bayes_nn, n_samples, lr, delta, train_loader1)
print('LangD initialized')

torch.save(langd.bayes_nn.nnets[0], "check.pth")

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
            np.save('log_like_loss.npy', loss)
    print('Training finished...')

    torch.save(langd.bayes_nn.nnets[0], "trained_model.pth")




# plt.semilogy( np.abs(log_post[int(epochs/10):epochs-1]) , 'k.')
# plt.title('Log posterior', fontname = 'Times New Roman', fontsize=12)


# data = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(un))
# train_loader2 = torch.utils.data.DataLoader(data, batch_size=500, shuffle=False)
# u_hat_all, f_hat_all = langd.compute(train_loader2)
# u_mean = torch.zeros(len(u_hat_all[0]))
# u_std = torch.zeros(len(u_hat_all[0]))
# f_mean = torch.zeros(len(u_hat_all[0]))
# f_std = torch.zeros(len(u_hat_all[0]))
# for i in range(len(u_hat_all[0])):
#     e = []
#     d = []
#     for j in range(len(u_hat_all)):
#         e.append(u_hat_all[j][i])
#         d.append(f_hat_all[j][i])
#     u_mean[i] = torch.tensor(e).mean()
#     u_std[i] = torch.tensor(e).std()
#     f_mean[i] = torch.tensor(d).mean()
#     f_std[i] = torch.tensor(d).std()
    

# plt.plot(xb2, ub, 'ro', label="data")
# plt.plot(x, u, 'b--', label="actual")
# plt.plot(x, u_mean, 'k-', label="predicted")
# plt.fill_between(x, u_mean - u_std, u_mean + u_std, color=[0.0,0.0,0.0], alpha=0.2)
# # plt.plot(x, u_mean + u_std, 'r-', x, u_mean - u_std, 'r-')
# plt.legend()
# plt.title('u(x)', fontname = 'Times New Roman', fontsize=12)
# plt.show()

# plt.plot(xb1, fb, 'ro', label="data")
# plt.plot(x, f, 'b--', label="actual")
# plt.plot(x, f_mean, 'k-', label="predicted")
# plt.fill_between(x, f_mean - f_std, f_mean + f_std, color=[0.0,0.0,0.0], alpha=0.2)
# # plt.plot(x, u_mean + u_std, 'r-', x, u_mean - u_std, 'r-')
# plt.title('f(x)', fontname = 'Times New Roman', fontsize=12)
# plt.legend()
# plt.show()

# sys.exit()

# mean_pred_k, std_pred_k = langd.compute_k()
# print("Mean value of k = ", mean_pred_k.item())
# print("SD value of k = ", std_pred_k.item())




# Results = {
#   "epochs": epochs,
#   "log_post": log_post,
#   "xb2": xb2,
#   "un": un,
#   "x": x,
#   "u": u,
#   "u_mean": u_mean,
#   "u_std": u_std,
#   "xb1": xb1,
#   "fb": fb,
#   "f": f,
#   "f_mean": f_mean,
#   "f_std": f_std}

# filename = '1D_ODE_results.npy'
# np.save(filename, Results)
