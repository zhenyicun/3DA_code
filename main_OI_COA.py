#%matplotlib inline
#%pylab inline
#pylab.rcParams['figure.figsize'] = (16, 9)

# analog data assimilation

import os
import numpy as np
from tqdm import tqdm
from common import *



def find_x_neighbor(lon,lat,LON,LAT,r_spat):
    idx = np.where(np.sqrt((lon-LON)**2 + (lat - LAT)**2) <= r_spat)[0]
    return idx

def find_t_neighbor(t,T,r_temp):
    idx = np.where(np.abs(t - T) < r_temp)[0]
    return idx

def find_obs_cylinder(lon_sa,lat_sa,time_sa,lonx,latx,tx,r_spat,r_temp):
    idx = np.where((np.sqrt((lon_sa - lonx)**2 + (lat_sa - latx)**2) < r_spat) & (np.abs(time_sa - tx) < r_temp))[0]
    return idx



def func_C_Le_Traon(dx,dt,s_spat,s_temp):
    a = 3.34/s_spat
    c = (1. + a*dx + 1./6.*(a*dx)**2. -1./6.*(a*dx)**3.) * np.exp(-a*dx) * np.exp(-dt**2./s_temp**2.)
    return c

def find_BH_Le_Traon(lon1,lat1,t1,lon2,lat2,t2,s_spat,s_temp):
    nx1 = len(lon1)
    nt1 = len(t1)
    
    lon1_tmp = np.repeat(lon1[np.newaxis],nt1,0).flatten()
    lat1_tmp = np.repeat(lat1[np.newaxis],nt1,0).flatten()
    t1_tmp = np.repeat(t1[np.newaxis].T,nx1,1).flatten()
    n1 = nx1*nt1


    n2 = len(lon2)

    dx = np.sqrt((np.repeat(lon2[np.newaxis],n1,0) - np.repeat(lon1_tmp[np.newaxis].T,n2,1))**2.0 + (np.repeat(lat2[np.newaxis],n1,0) - np.repeat(lat1_tmp[np.newaxis].T,n2,1))**2.0)

    dt = np.abs(np.repeat(t2[np.newaxis],n1,0) - np.repeat(t1_tmp[np.newaxis].T,n2,1))

    BH = func_C_Le_Traon(dx,dt,s_spat,s_temp)


    return BH

def find_HBH_Le_Traon(lon1,lat1,t1,lon2,lat2,t2,s_spat,s_temp):
    n1 = len(lon1)    
    n2 = len(lon2)

    dx = np.sqrt((np.repeat(lon2[np.newaxis],n1,0) - np.repeat(lon1[np.newaxis].T,n2,1))**2.0 + (np.repeat(lat2[np.newaxis],n1,0) - np.repeat(lat1[np.newaxis].T,n2,1))**2.0)

    dt = np.abs(np.repeat(t2[np.newaxis],n1,0) - np.repeat(t1[np.newaxis].T,n2,1))

    HBH = func_C_Le_Traon(dx,dt,s_spat,s_temp)

    return HBH


cwd = os.getcwd()
path_for_save = cwd + '/data/results/OI_COA/'



path_OCCIPUT = cwd + '/data/OCCIPUT/'
U = np.load(path_OCCIPUT + 'U.npy')
S = np.load(path_OCCIPUT + 'S.npy')
idx_my_region = np.load(path_OCCIPUT + 'idx_my_region.npy')
ssh_mean = np.load(path_OCCIPUT + 'ssh_mean.npy')
lon_grid = np.load(path_OCCIPUT + 'lon_OCCIPUT.npy')
lat_grid = np.load(path_OCCIPUT + 'lat_OCCIPUT.npy')
lon = lon_grid.flatten()
lat = lat_grid.flatten()

cov_ssh = U.dot(np.diag(S)).dot(U.T)
var_ssh = np.diag(cov_ssh)

lon1 = lon[idx_my_region]
lat1 = lat[idx_my_region]

path_obs = cwd + '/data/obs/'
lon_at = np.load(path_obs + 'lon_at.npy')
lat_at = np.load(path_obs + 'lat_at.npy')
time_at = np.load(path_obs + 'time_at.npy')
ssh_at = np.load(path_obs + 'ssh_at.npy')
ssh_true = np.load(path_obs + 'ssh_true.npy')
time_true = np.load(path_obs + 'time_true.npy')
Loc_H = np.load(path_obs + 'Loc_H.npy')



################################################################### OI Parameters #########################################################################################
r_spat = 1.5
s_spat = 1.5
r_temp = 365.
s_temp = 20.
R = 0.0035



################################################################################################################################################################
print "================================ OI start ======================================"
time_xs = np.copy(time_true)
ssh_xs = np.repeat(ssh_mean[np.newaxis],ssh_true.shape[0],0)
var_xs = np.zeros(ssh_xs.shape)
for ix in tqdm(range(ssh_true.shape[1])):
    lon_tmp = lon1[ix];lat_tmp = lat1[ix]
    ix_neighbor_tmp = find_x_neighbor(lon_tmp,lat_tmp,lon1,lat1,r_spat)
    ix_pos = np.where(ix_neighbor_tmp == ix)[0][0]

    it_neighbor_tmp = find_t_neighbor(time_true[0], time_true, 366)
    i_at_tmp = find_obs_cylinder(lon_at,lat_at,time_at,lon_tmp,lat_tmp,time_true[0],r_spat,366.0)
    if len(i_at_tmp) == 0:
        continue
    xb_tmp = ssh_mean[ix]*np.ones(365)

    lon_at_tmp = lon_at[i_at_tmp]
    lat_at_tmp = lat_at[i_at_tmp]
    ssh_at_tmp = ssh_at[i_at_tmp]
    time_at_tmp = time_at[i_at_tmp]
    H_tmp = get_H(lon_at_tmp,lat_at_tmp,lon_grid,lat_grid,idx_my_region,Loc_H)

    BH_tmp = find_BH_Le_Traon(lon1[ix:ix+1],lat1[ix:ix+1], time_true[it_neighbor_tmp], lon_at_tmp,lat_at_tmp,time_at_tmp,s_spat,s_temp)
    HBH_tmp = find_HBH_Le_Traon(lon_at_tmp,lat_at_tmp,time_at_tmp,lon_at_tmp,lat_at_tmp,time_at_tmp,s_spat,s_temp)

    rootvar1_tmp = np.sqrt(var_ssh[ix])*np.ones(365)
    rootvar2_tmp = np.sqrt(np.sum(H_tmp.dot(np.diag(var_ssh)), 1))

    BH_tmp = (rootvar1_tmp*BH_tmp.T).T*rootvar2_tmp
    HBH_tmp = (rootvar2_tmp*HBH_tmp.T).T*rootvar2_tmp		
    K = BH_tmp.dot(np.linalg.inv(HBH_tmp+R*np.eye(HBH_tmp.shape[0])))
    yb = H_tmp.dot(ssh_mean)

    xs_tmp = xb_tmp + K.dot(ssh_at_tmp - yb)
    ssh_xs[:,ix] = np.copy(xs_tmp)
    var_xs[:,ix] = rootvar1_tmp**2 - np.diag(K.dot(BH_tmp.T))
RMSE = np.sqrt(np.mean((ssh_xs - ssh_true)**2))


print r_spat,s_spat,r_temp,s_temp,R,RMSE

np.save(path_for_save + 'ssh_xs', ssh_xs)
np.save(path_for_save + 'ssh_true', ssh_true)
np.save(path_for_save + 'time_xs',time_xs)
np.save(path_for_save + 'time_true',time_true)
np.save(path_for_save + 'var_xs',var_xs)

