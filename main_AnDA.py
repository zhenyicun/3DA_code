
import os
import pylab
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.linalg import sqrtm
from datetime import date
from tqdm import tqdm
from common import *



cwd = os.getcwd()

######################## parameters ###############################
path_for_save = cwd+'/data/results/AnDA/'
max_mode = 100
do_KS = 'yes'
Ne = 1000
R = 0.0004
do_localization = 'no'
rloc = 10.
AF_k = 1000
###################################################################


path_OCCIPUT = cwd+'/data/OCCIPUT/'
U = np.load(path_OCCIPUT + 'U.npy')
S = np.load(path_OCCIPUT + 'S.npy')
idx_my_region = np.load(path_OCCIPUT + 'idx_my_region.npy')
ssh_mean = np.load(path_OCCIPUT + 'ssh_mean.npy')
lon_grid = np.load(path_OCCIPUT + 'lon_OCCIPUT.npy')
lat_grid = np.load(path_OCCIPUT + 'lat_OCCIPUT.npy')
lon = lon_grid.flatten()
lat = lat_grid.flatten()

path_obs = cwd+'/data/obs/'
lon_at = np.load(path_obs + 'lon_at.npy')
lat_at = np.load(path_obs + 'lat_at.npy')
time_at = np.load(path_obs + 'time_at.npy')
ssh_at = np.load(path_obs + 'ssh_at.npy')
ssh_true = np.load(path_obs + 'ssh_true.npy')
time_true = np.load(path_obs + 'time_true.npy')
Loc_H = np.load(path_obs + 'Loc_H.npy')

path_catalog = cwd+'/data/catalog/'
analogs = np.load(path_catalog + 'analogs.npy')
successors = np.load(path_catalog + 'successors.npy')


################################################## construct initial x from obs ###################################################################################
idx_tmp = np.where((time_at >= time_true[0]-15.) & (time_at < time_true[0] + 15.))[0]

if len(idx_tmp) == 0:
    print("error, choose larger time gap.")
    stop
lon_at_tmp = lon_at[idx_tmp]
lat_at_tmp = lat_at[idx_tmp]
ssh_at_tmp = ssh_at[idx_tmp]	
H_tmp =  get_H(lon_at_tmp,lat_at_tmp,lon_grid,lat_grid,idx_my_region,Loc_H)
tmp = ssh_at_tmp - H_tmp.dot(ssh_mean)
HU = H_tmp.dot(U[:,0:max_mode])
ssh_pc_tmp = np.linalg.inv(HU.T.dot(HU)).dot(HU.T).dot(tmp)
x_init = np.copy(ssh_pc_tmp)


################################################## AnDA Preparation ###################################################################################

class catalog:
    analogs = np.copy(analogs)
    successors = np.copy(successors)
class AF:
    catalog = catalog
    k = AF_k
    modes_analog = np.arange(max_mode)

print "use R = ", R, ', k=',AF.k,', Ne =',Ne
################################################# Data Assimilation ####################################################################################
print "Start Data Assimilation"
xfmean = np.zeros((365,max_mode))
xfens = np.zeros((365,Ne,max_mode))
xpens = np.zeros((365,Ne,max_mode))
xamean = np.zeros((365,max_mode))
xaens = np.zeros((365,Ne,max_mode))
Pf = np.zeros((365,max_mode,max_mode))
Pa = np.zeros((365,max_mode,max_mode))
 
xfmean_tmp = np.copy(x_init)
Pf_tmp = np.eye(max_mode)
xfens_tmp = np.random.multivariate_normal(xfmean_tmp, Pf_tmp, Ne)
xpens_tmp = np.copy(xfens_tmp)
time_xf = np.copy(time_true)
time_xa = np.copy(time_true)
time_xs = np.copy(time_true)

kdt = KDTree(AF.catalog.analogs, leaf_size=50, metric='euclidean')
if do_localization == 'yes':
    Cloc = get_Cloc(rloc,lon[idx_my_region],lat[idx_my_region])

t_now = date(2004,1,1).toordinal()
t_min = date(2004,1,1).toordinal() 
t_max = t_min+365.

for t_now in tqdm(time_true):
    i_now = np.where(time_true == t_now)[0][0]

    xfmean[i_now] = np.copy(xfmean_tmp)
    xfens[i_now] = np.copy(xfens_tmp)
    xpens[i_now] = np.copy(xpens_tmp)
    Pf[i_now] = np.copy(Pf_tmp)

    idx_tmp = np.where((time_at >= t_now-0.5) & (time_at < t_now + 0.5))[0]    
    if len(idx_tmp) > 0:
	lon_at_tmp = lon_at[idx_tmp]
	lat_at_tmp = lat_at[idx_tmp]
	ssh_at_tmp = ssh_at[idx_tmp]	

	H_tmp =  get_H(lon_at_tmp,lat_at_tmp,lon_grid,lat_grid,idx_my_region,Loc_H)

	yo_tmp = ssh_at_tmp - H_tmp.dot(ssh_mean)
	yoens_tmp = np.repeat(yo_tmp[np.newaxis], Ne, 0)

	H_DA_tmp = H_tmp.dot(U[:,0:max_mode])

	K = Pf_tmp.dot(H_DA_tmp.T).dot(np.linalg.inv(R*np.eye(len(idx_tmp)) + (H_DA_tmp.dot(Pf_tmp.dot(H_DA_tmp.T)))))
	eta = np.random.normal(0.0, np.sqrt(R), (Ne,len(idx_tmp)))
	eta = eta - np.repeat(np.mean(eta, 0)[np.newaxis],   Ne, 0)	

	xaens_tmp = xfens_tmp +(yoens_tmp + eta - xfens_tmp.dot(H_DA_tmp.T)).dot(K.T)

	Pa_tmp = (xaens_tmp - np.repeat(np.mean(xaens_tmp,0)[np.newaxis], Ne, 0)).T.dot(xaens_tmp - np.repeat(np.mean(xaens_tmp,0)[np.newaxis], Ne, 0))/(Ne - 1.)
	Pa_tmp = (Pa_tmp + Pa_tmp.T)/2.
	xamean_tmp = np.mean(xaens_tmp,0)
        xamean[i_now] = np.copy(xamean_tmp)
	xaens[i_now] = np.copy(xaens_tmp)
	Pa[i_now] = np.copy(Pa_tmp)

	Pf_tmp = np.copy(Pa_tmp)
	xfens_tmp = np.copy(xaens_tmp)
	xfmean_tmp = np.copy(xamean_tmp)
    else:
        xamean[i_now] = np.copy(xfmean_tmp)
        xaens[i_now] = np.copy(xfens_tmp)
        Pa[i_now] = np.copy(Pf_tmp)

    #################################### analog forecast #############################################################
    dist_knn, index_knn = kdt.query(xfens_tmp, AF.k)
    analogs_tmp = np.zeros((Ne,AF.k,AF.catalog.analogs.shape[-1]))
    successors_tmp = np.zeros((Ne,AF.k,AF.catalog.successors.shape[-1]))
    for i_N in range(Ne):
        analogs_tmp[i_N] = np.copy(AF.catalog.analogs[index_knn[i_N]])
        successors_tmp[i_N] = np.copy(AF.catalog.successors[index_knn[i_N]])
    xfens_tmp,xpens_tmp = Analog_forecast2(xfens_tmp,analogs_tmp,successors_tmp,dist_knn)
    ##################################################################################################################
    xfmean_tmp = np.mean(xfens_tmp, 0)
    Pf_tmp = (xfens_tmp - np.repeat(np.mean(xfens_tmp,0)[np.newaxis], Ne, 0)).T.dot(xfens_tmp - np.repeat(np.mean(xfens_tmp,0)[np.newaxis], Ne, 0))/(Ne -1.)
    Pf_tmp = (Pf_tmp + Pf_tmp.T)/2.0
    if do_localization == 'yes':
        Pf_tmp = get_Bloc(Pf_tmp,U[:,0:max_mode],Cloc)
    


var_xs = np.zeros((365,U.shape[0]))
xsmean = np.copy(xfmean)
xsens = np.copy(xaens)

if do_KS == 'yes':
    print "Start Kalman Smoother"
    t_now = t_max-1

    for t_now in tqdm(np.flip(time_xs[0:-1])):
	i_now = np.where(time_xs == t_now)[0][0]

	x1 = np.copy(xaens[i_now,:,:])
	x11 = np.copy(xpens[i_now+1,:,:])
	x2 = np.copy(xfens[i_now+1,:,:])
	xens_tmp = np.copy(xaens[i_now,:,:])

        if do_localization == 'yes':
            B11 = sample_cov(x1,x1)
            B2 = sample_cov(x2,x2)
            x1pert = x1 - np.repeat(np.mean(x1,0)[np.newaxis], Ne, 0)
            x11pert = x11 - np.repeat(np.mean(x11,0)[np.newaxis], Ne, 0)
            x1pertpinv = np.linalg.pinv(x1pert)
            At = x1pertpinv.dot(x11pert)
            B11 = get_Bloc(B11,U[:,0:max_mode],Cloc)
            B2 = get_Bloc(B2,U[:,0:max_mode],Cloc)
            B1A = B11.dot(At)
        elif do_localization == 'no':
            B1A = sample_cov(x1,x11)
	    B2 = sample_cov(x2,x2)

	J = B1A.dot(np.linalg.inv(B2))
	xsens[i_now,:,:] = xens_tmp + (xsens[i_now+1,:,:] - xfens[i_now+1,:,:]).dot(J.T)
	xsmean[i_now,:] = np.mean(xsens[i_now,:,:],0)


    for i in range(365):
        var_xs[i] = np.diag(U[:,0:max_mode].dot(np.cov(xsens[i].T)).dot(U[:,0:max_mode].T))

#################################################### reconstruction for EOFs ####################################################################################


ssh_xf = np.repeat(ssh_mean[np.newaxis],xfmean.shape[0],0) + xfmean.dot(U[:,0:max_mode].T)
ssh_xs = np.repeat(ssh_mean[np.newaxis],xsmean.shape[0],0) + xsmean.dot(U[:,0:max_mode].T)
ssh_xa = np.repeat(ssh_mean[np.newaxis],xamean.shape[0],0) + xamean.dot(U[:,0:max_mode].T)

print 'use R = ',R,', max_mode = ',max_mode, ', Ne = ', Ne,', AF.k=',AF.k,', RMSE: ', np.sqrt(np.mean((ssh_xs - ssh_true)**2))

np.save(path_for_save + 'ssh_xf',ssh_xf)
np.save(path_for_save + 'ssh_xa',ssh_xa)
np.save(path_for_save + 'ssh_xs',ssh_xs)
np.save(path_for_save + 'ssh_true',ssh_true)
np.save(path_for_save + 'var_xs',var_xs)
np.save(path_for_save + 'max_mode',max_mode)
np.save(path_for_save + 'Loc_H',Loc_H)
np.save(path_for_save + 'time_xf',time_xf)
np.save(path_for_save + 'time_xa',time_xa)
np.save(path_for_save + 'time_xs',time_xs)
np.save(path_for_save + 'time_true',time_true)

