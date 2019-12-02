
import numpy as np
from scipy.linalg import sqrtm



def get_i_good(lon_sa,lat_sa,lon_grid,lat_grid,idx_my_region,L):
    i_good = np.array([])
    lon_g = lon_grid.flatten()[idx_my_region]
    lat_g = lat_grid.flatten()[idx_my_region]

    for i_sa in range(lon_sa.shape[0]):
	idx_min = np.argmin((lon_grid.flatten() - lon_sa[i_sa])**2 + (lat_grid.flatten() - lat_sa[i_sa])**2)
	if (idx_min in idx_my_region) and len(np.where((lon_g < lon_sa[i_sa] + L) & (lon_g > lon_sa[i_sa] - L) & (lat_g < lat_sa[i_sa] + L) & (lat_g > lat_sa[i_sa] - L))[0]) > 0:  
	    i_good = np.hstack((i_good, i_sa))
    return i_good

def get_H(lon,lat,lon_grid,lat_grid,idx_my_region,L):
    lon_grid_tmp = lon_grid.flatten()[idx_my_region]
    lat_grid_tmp = lat_grid.flatten()[idx_my_region]

    n_obs = len(lon)
    H = np.zeros((n_obs,len(idx_my_region)))
    for i in range(n_obs):
	lon_tmp = lon[i]
	lat_tmp = lat[i]
	
	dist_tmp = np.sqrt((lon_grid_tmp - lon_tmp)**2 + (lat_grid_tmp - lat_tmp)**2)
	idx_tmp = np.argsort(dist_tmp)[0:4]

	wt_tmp = np.exp(-(dist_tmp[idx_tmp]**2 - np.min(dist_tmp[idx_tmp])**2)/L**2)
	H[i,idx_tmp] = wt_tmp/np.sum(wt_tmp)
    return H

def mk_stochastic(T):
    """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

    if len(T.shape) == 1:
        T = normalise(T);
    else:
        n = len(T.shape);
        # Copy the normaliser plane for each i.
        normaliser = np.sum(T,n-1);
        normaliser = np.dstack([normaliser]*T.shape[n-1])[0];
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0

        normaliser = normaliser + 1*(normaliser==0);
        T = T/normaliser.astype(float);
    return T



def Analog_forecast2(xens, analogs,successors,dist_knn):
    
    Ne, n = xens.shape
    xfens = np.zeros([Ne,n])
    xfens_mean = np.zeros([Ne,n])

    lambdaa = np.median(dist_knn)

    weights = mk_stochastic(np.exp(-np.power(dist_knn,2)/lambdaa))
        

    for i_N in range(0,Ne):

        X = analogs[i_N]                
        Y = successors[i_N]                
        w = weights[i_N,:][np.newaxis]
                
        Xm = np.array([np.sum(X*np.repeat(w.T,n,1),0)])
        Xc = X - np.repeat(Xm,np.shape(X)[0],0)
                
        U,S,V = np.linalg.svd(Xc,full_matrices=False)
        ind = np.where(S/np.sum(S)>0.01)[0] # keep eigen values higher than 1%
                
        Xr = np.hstack((np.ones([np.shape(X)[0],1]),np.dot(Xc,V.T[:,ind])))
        Cxx = np.dot(np.repeat(np.array([w.flatten()]).T,np.shape(Xr)[1],1).T*Xr.T,Xr)
        Cxx2 = np.dot(np.repeat(np.array([(w**2).flatten()]).T,np.shape(Xr)[1],1).T*Xr.T,Xr)
        Cxy = np.dot(np.repeat(np.array([w.flatten()]).T,np.shape(Y)[1],1).T*Y.T,Xr)
        inv_Cxx = np.linalg.inv(Cxx) # in case of error here, increase the number of analogs (AF.k option)
        beta = np.dot(inv_Cxx,Cxy.T)
        X0 = xens[i_N,:]-Xm
        X0r = np.hstack((np.ones([np.shape(X0)[0],1]),np.dot(X0,V.T[:,ind])))

        xfens_mean[i_N] = np.dot(X0r,beta)
        pred = np.dot(Xr,beta)
        res = Y-pred                

        cov_xfc = np.dot(np.repeat(np.array([w.flatten()]).T,np.shape(res)[1],1).T*res.T,res)/(1-np.trace(np.dot(Cxx2,inv_Cxx)))
        cov_xf = cov_xfc*(1+np.trace(Cxx2.dot(inv_Cxx).dot(X0r.T).dot(X0r).dot(inv_Cxx)))

        xfens[i_N] = np.random.multivariate_normal(xfens_mean[i_N],cov_xf)
   
    return xfens, xfens_mean



def get_Bloc(B,U,Cloc):
    B1 = U.dot(B).dot(U.T)
    B1 = B1*Cloc
    B2 = U.T.dot(B1).dot(U)
    return B2

def get_Cloc(ROI,lon,lat):
    n = lon.shape[0]
    lonlon = np.outer(lon,np.ones(n))
    latlat = np.outer(np.ones(n),lat)
    dd = (lonlon - lonlon.T)**2. + (latlat.T - latlat)**2.
    Cloc = np.exp(-dd/ROI**2.)
    return Cloc

def sample_cov(Xens1,Xens2):
    Ne,n1 = Xens1.shape
    xmean1 = np.mean(Xens1,0)
    Xpert1 = Xens1 - np.repeat(xmean1[np.newaxis],Ne,0)

    Ne,n2 = Xens2.shape
    xmean2 = np.mean(Xens2,0)
    Xpert2 = Xens2 - np.repeat(xmean2[np.newaxis],Ne,0)

    B = Xpert1.T.dot(Xpert2)/(Ne-1.)
    return B


