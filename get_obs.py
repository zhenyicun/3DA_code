
import os,sys
sys.path.append('/home/zhenyicun/AnDA/satellite/codes/')
import pylab
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.linalg import sqrtm
from datetime import date

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from common import *

cwd = os.getcwd()
path_for_save = cwd+'/data/obs/'
Loc_H = 0.15
R_obs = 0.0

path_OCCIPUT = cwd+'/data/OCCIPUT/'
U = np.load(path_OCCIPUT + 'U.npy')
idx_my_region = np.load(path_OCCIPUT + 'idx_my_region.npy')
ssh_mean = np.load(path_OCCIPUT + 'ssh_mean.npy')
lon_grid = np.load(path_OCCIPUT + 'lon_OCCIPUT.npy')
lat_grid = np.load(path_OCCIPUT + 'lat_OCCIPUT.npy')
ssh1 = np.load(path_OCCIPUT + 'ssh_1.npy')
lon = lon_grid.flatten()
lat = lat_grid.flatten()


path_obs = cwd + '/data/obs/'
lon_at = np.load(path_obs + 'lon_at.npy')
lat_at = np.load(path_obs + 'lat_at.npy')
time_at = np.load(path_obs + 'time_at.npy')

ssh_true = np.copy(ssh1[-365:])
t_start = date(2004,1,1).toordinal()
time_true = (np.arange(365) + t_start).astype(int)

ssh_at = np.zeros(lon_at.shape)

for t in time_true:
    i_true = np.where(time_true == t)[0][0]
    idx_tmp = np.where((time_at >= t-0.5) & (time_at < t + 0.5))[0]
    if len(idx_tmp) > 0:
	lon_at_tmp = lon_at[idx_tmp]
	lat_at_tmp = lat_at[idx_tmp]
	H_tmp =  get_H(lon_at_tmp,lat_at_tmp,lon_grid,lat_grid,idx_my_region,Loc_H)

	ssh_at[idx_tmp] = H_tmp.dot(ssh_true[i_true]) + np.random.normal(0,np.sqrt(R_obs),len(idx_tmp))


np.save(path_for_save + 'ssh_at',ssh_at)
np.save(path_for_save + 'ssh_true',ssh_true)
np.save(path_for_save + 'time_true',time_true)
np.save(path_for_save + 'Loc_H',Loc_H)

