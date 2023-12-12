#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:46:42 2023

@author: kevin
"""

import scipy.io as sp
import numpy as np
import sciann as sn 
import matplotlib.pyplot as plt 
import os
import netCDF4 as nc
import pickle


def PrepareData(ts = 100, xp=40, yp=40, zp=40):
    Data = sp.loadmat('sbl_coarse.mat')
    
    inv_x = int(40/xp)
    inv_y = int(40/yp)
    inv_z = int(40/zp)
    inv_t = int(3600/ts)
    
    u = Data['u'][0:3599:inv_t, 0:39:inv_z, 0:39:inv_y, 0:39:inv_x]
    v = Data['v'][0:3599:inv_t, 0:39:inv_z, 0:39:inv_y, 0:39:inv_x]
    w = Data['w'][0:3599:inv_t, 0:39:inv_z, 0:39:inv_y, 0:39:inv_x]
    x = Data['x'][:,0:39:inv_x]
    y = Data['y'][:,0:39:inv_y]
    z = Data['z'][:,0:39:inv_z]
    t = Data['time'][:,0:3599:inv_t]
    
    x = np.reshape(x, [x.shape[1],1])
    y = np.reshape(y, [y.shape[1],1])
    z = np.reshape(z, [z.shape[1],1])
    t = np.reshape(t, [t.shape[1],1])
    
    co = 0
    dim = ts*xp*yp*zp
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
    
    return (xt,yt,zt,tt,ut,vt,wt)


#%%

Training = True

ts = 50
xp = 10
yp = 10
zp = 10
T_train = ts*xp*yp*zp
x_train, y_train, z_train, t_train, u_train, v_train, w_train = PrepareData(ts, xp, yp, zp)


x = sn.Variable("x", dtype='float32')
y = sn.Variable("y", dtype='float32')
z = sn.Variable("z", dtype='float32')
t = sn.Variable("t", dtype='float32')

P = sn.Functional("P", [x, y, z, t], 2*[20], 'tanh')
u = sn.Functional("u", [x, y, z, t], 2*[20], 'tanh')
v = sn.Functional("v", [x, y, z, t], 2*[20], 'tanh')
w = sn.Functional("w", [x, y, z, t], 2*[20], 'tanh')

Re = sn.Parameter(1e2, inputs=[x,y,z,t], name="Re")
# lambda1 = sn.Parameter(0.0, inputs=[x,y,t], name="lambda1")
# lambda2 = sn.Parameter(0.0, inputs=[x,y,t], name="lambda2")
# th1 = sn.Parameter(0.0, inputs=[x,y,t], name="th1")
# th2 = sn.Parameter(0.0, inputs=[x,y,t], name="th2")

u_t = sn.diff(u, t)
u_x = sn.diff(u, x)
u_y = sn.diff(u, y)
u_z = sn.diff(u, z)
u_xx = sn.diff(u, x, order=2)
u_yy = sn.diff(u, y, order=2)
u_zz = sn.diff(u, z, order=2)

v_t = sn.diff(v, t)
v_x = sn.diff(v, x)
v_y = sn.diff(v, y)
v_z = sn.diff(v, z)
v_xx = sn.diff(v, x, order=2)
v_yy = sn.diff(v, y, order=2)
v_zz = sn.diff(v, z, order=2)

w_t = sn.diff(w, t)
w_x = sn.diff(w, x)
w_y = sn.diff(w, y)
w_z = sn.diff(w, z)
w_xx = sn.diff(w, x, order=2)
w_yy = sn.diff(w, y, order=2)
w_zz = sn.diff(w, z, order=2)

P_x = sn.diff(P, x)
P_y = sn.diff(P, y)
P_z = sn.diff(P, z)

D1 = sn.Data(u)
D2 = sn.Data(v)
D3 = sn.Data(w)

L1 = u_t + u*u_x + v*u_y + w*u_z + P_x - (u_xx + u_yy + u_zz)/Re
L2 = v_t + u*v_x + v*v_y + w*v_z + P_y - (v_xx + v_yy + v_zz)/Re
L3 = w_t + u*w_x + v*w_y + w*w_z + P_z - (w_xx + w_yy + w_zz)/Re
L4 = u_x + v_y + w_z
L5 = P*0.0

model = sn.SciModel(inputs=[x, y, z, t], targets=[D1, D2, D3, L1, L2, L3, L4, L5], loss_func="mse")
input_data = [x_train, y_train, z_train, t_train]

data_points = int(T_train / 10)
data_ind = np.random.choice(T_train, data_points, replace=False).reshape([data_points,1])
data_D1 = u_train[data_ind,0]
data_D2 = v_train[data_ind,0]
data_D3 = w_train[data_ind,0]
data_L1 = 'zeros'
data_L2 = 'zeros'
data_L3 = 'zeros'
data_L4 = 'zeros'
data_L5 = 'zeros'

target_data = [(data_ind, data_D1), (data_ind, data_D2), (data_ind, data_D3), data_L1, data_L2, data_L3, data_L4, data_L5]


if Training:
    history = model.train(
        x_true=input_data,
        y_true=target_data,
        epochs=5000, # 20000,
        batch_size=4096,
        shuffle=True,
        learning_rate=0.001,
        reduce_lr_after=100,
        stop_loss_value=1e-8,
        verbose=1
    )
    
    # with open('mytrain_NS_new.pickle', 'wb') as f:
    #         pickle.dump(history.history, f)
    
    model.save_weights('NS_PINN_weights.hdf5')

#%%

model.load_weights('NS_PINN_weights.hdf5')

plt.close('all')
tf = 25

Data = sp.loadmat('sbl_coarse.mat')
x = np.reshape(Data['x'], [40,])
y = np.reshape(Data['y'], [40,])
z = np.reshape(Data['z'], [40,])
t = np.reshape(Data['time'], [3600,])
x_data, y_data, z_data, t_data = np.meshgrid(x,y,z,t,)
u_pred = u.eval(model,[x_data, y_data, z_data, t_data])
v_pred = v.eval(model,[x_data, y_data, z_data, t_data])

# x_data, y_data, z_data, t_data = np.meshgrid(
#         np.linspace(0, 40, 50),
#         np.linspace(0, 40, 50),
#         np.linspace(0, 40, 50),
#         np.linspace(0, 3600, 50),)
# u_pred = u.eval(model,[x_data, y_data, z_data, t_data])
# v_pred = v.eval(model,[x_data, y_data, z_data, t_data])
  
    
fig = plt.figure(figsize=(4, 4))
s = 29
plt.pcolor(x_data[:,:,0,s], y_data[:,:,0,s], v_pred[:,:,0,s], cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.clim(2,8)
plt.colorbar()
    
fig = plt.figure(figsize=(4, 4))
#for j in range(32,32):
j = 3 + s
if j >= 10:
    F = 'cm1out_0000' + str(j) + '.nc'
else:
    F = 'cm1out_00000' + str(j) + '.nc'

F = os.path.join(os.getcwd(), 'ncfiles', F)
ds = nc.Dataset(F)
Ut = ds['uinterp'][:]
ut = Ut[:,5,:,:]
Pt = ds['prs'][:]
plt.pcolor(x_data[:,:,-1], y_data[:,:,-1],  ut[0,:,:], cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.clim(2,8)
plt.colorbar()

fig = plt.figure(figsize=(4,4))
err = abs(ut[0,:,:] - c_pred[:,:,s])/ut[0,:,:]
plt.pcolor(x_data[:,:,s], y_data[:,:,s], err, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.clim(0,1)
plt.colorbar()
plt.title('max perc error: ' + str(np.max(err)*100))

#%%
fig = plt.figure(figsize=(4, 4))
#for j in range(32,32):
pt = Pt[:,5,:,:]
plt.pcolor(x_data[:,:,-1], y_data[:,:,-1],  pt[0,:,:], cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
#plt.clim(2,8)
plt.colorbar()


