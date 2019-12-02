#%pylab inline
#%matplotlib inline

import os
import pylab
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib import cm
#from module_RSS_SST import *
#from module_AVISO_SSH import *
#from module_CORIOLIS import *
#from module_image import *
import module_image as mi
#from module_SST_SSH import *
import pylab
from tqdm import tqdm
from matplotlib.patches import Polygon
from datetime import date
cwd = os.getcwd()
pylab.rcParams['figure.figsize'] = (5, 5)




def my_month(n, leap_year = 'no'):#starting from 0
    if leap_year == 'no':
	nn = np.array([30,58,89,119,150,180,211,242,272,303,333,364])
    elif leap_year == 'yes':
	nn = np.array([30,59,90,120,151,181,212,243,273,304,334,365])
    else:
	nn = np.array([30,58,89,119,150,180,211,242,272,303,333,364])

    if n <=nn[0]:
	m1 = "Jan. ";m2 = 1
    elif n <=nn[1]:
	m1 = "Feb. ";m2 = 2
    elif n <=nn[2]:	
	m1 = "Mar. ";m2 = 3
    elif n <=nn[3]:	
	m1 = "Apr. ";m2 = 4
    elif n <=nn[4]:
	m1 = "May  ";m2 = 5
    elif n <=nn[5]:
	m1 = "June ";m2 = 6
    elif n <=nn[6]:
	m1 = "July ";m2 = 7
    elif n <=nn[7]:
	m1 = "Aug. ";m2 = 8
    elif n <=nn[8]:
	m1 = "Sep. ";m2 = 9
    elif n <=nn[9]:
	m1 = "Oct. ";m2 = 10
    elif n <=nn[10]:
	m1 = "Nov. ";m2 = 11
    elif n <=nn[11]:
	m1 = "Dec. ";m2 = 12
    return m1,m2

def my_day(n, leap_year = 'no'):#starting from 0
    if leap_year == 'no':
	nn = np.array([30,58,89,119,150,180,211,242,272,303,333,364])
    elif leap_year == 'yes':
	nn = np.array([30,59,90,120,151,181,212,243,273,304,334,365])
    else:
	nn = np.array([30,58,89,119,150,180,211,242,272,303,333,364])

    if n <=nn[0]:
	d = n+1
    elif n <=nn[1]:
	d = n - nn[0]
    elif n <=nn[2]:	
	d = n - nn[1]
    elif n <=nn[3]:	
	d = n - nn[2]
    elif n <=nn[4]:
	d = n - nn[3]
    elif n <=nn[5]:
	d = n - nn[4]
    elif n <=nn[6]:
	d = n - nn[5]
    elif n <=nn[7]:
	d = n - nn[6]
    elif n <=nn[8]:
	d = n - nn[7]
    elif n <=nn[9]:
	d = n - nn[8]
    elif n <=nn[10]:
	d = n - nn[9]
    elif n <=nn[11]:
	d = n - nn[10]

    return d
    
def month_str(m):
    if m==1:
	m1 = "Jan. "
    elif m==2:
	m1 = "Feb. "
    elif m==3:	
	m1 = "Mar. "
    elif m==4:
	m1 = "Apr. "
    elif m==5:
	m1 = "May  "
    elif m==6:
	m1 = "June "
    elif m==7:
	m1 = "July "
    elif m==8:
	m1 = "Aug. "
    elif m==9:
	m1 = "Sep. "
    elif m==10:
	m1 = "Oct. "
    elif m==11:
	m1 = "Nov. "
    elif m==12:
	m1 = "Dec. "
    return m1

def day_str(d):
    if d < 10:
	return str(d)+' '
    else:
	return str(d)

def draw_screen_poly(lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', alpha=0.4, edgecolor='black' )
    plt.gca().add_patch(poly)

def mask_map(X,idx,myshape):
    Xout = np.zeros(myshape).flatten()
    Xout[idx] = np.copy(X)
    Xout = ma.masked_values(Xout,0)
    Xout = np.reshape(Xout,myshape)
    return Xout

def get_obs_spatial_density(lon_at,lat_at,lon_grid,lat_grid):
    dens = np.zeros(lon_grid.shape)
    L = 0.2
    for i in range(lon_at.shape[0]):
        dens = dens + np.exp(-((lon_grid-lon_at[i])**2.+(lat_grid - lat_at[i])**2.)/L**2.)
    return dens
        
        
d=10
vmin = -0.4
vmax = 1.2
vmin1 = 0.
vmax1 = 1.35
blues = cm.Blues(np.linspace(1,0, num=1500))
reds = cm.Reds(np.linspace(0,1, num=1500))
colors = np.vstack((blues, reds))
mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

colors1 = np.copy(reds)
mycmap1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors1)

path_OCCIPUT = cwd + '/data/OCCIPUT/'
U = np.load(path_OCCIPUT + 'U.npy')
S = np.load(path_OCCIPUT + 'S.npy')
idx_my_region = np.load(path_OCCIPUT + 'idx_my_region.npy')
idx_FLcoast = np.load(path_OCCIPUT + 'idx_FLcoast.npy')
idx_YCcoast = np.load(path_OCCIPUT + 'idx_YCcoast.npy')
ssh_mean = np.load(path_OCCIPUT + 'ssh_mean.npy')
lon_OCCIPUT = np.load(path_OCCIPUT + 'lon_OCCIPUT.npy')
lat_OCCIPUT = np.load(path_OCCIPUT + 'lat_OCCIPUT.npy')


var_clim = np.diag(U.dot(np.diag(S)).dot(U.T))
varclim_map = np.zeros(41*41)
varclim_map[idx_my_region] = np.copy(var_clim)
varclim_map = np.reshape(varclim_map,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
varclim_map = ma.masked_values(varclim_map,0)





path_data_AnDA = cwd + '/data/results/AnDA_k1000_ROI10/'
ssh_AnDA = np.load(path_data_AnDA + "ssh_xs.npy")
time_AnDA = np.load(path_data_AnDA + "time_xs.npy")
var_xs_AnDA = np.load(path_data_AnDA + 'var_xs.npy')

path_data_OI = cwd + '/data/results/OI_6/'
ssh_OI = np.load(path_data_OI + "ssh_xs.npy")
time_OI = np.copy(time_AnDA)
var_xs_OI = np.load(path_data_OI + 'var_xs.npy')

path_data_COA = cwd + '/data/results/OI_COA/'
ssh_COA = np.load(path_data_COA + "ssh_xs.npy")
time_COA = np.copy(time_AnDA)
var_xs_COA = np.load(path_data_COA + 'var_xs.npy')



path_data_obs = cwd + '/data/obs/'
ssh_sa = np.load(path_data_obs + "ssh_at.npy")
lon_sa  = np.load(path_data_obs + "lon_at.npy")
lat_sa  = np.load(path_data_obs + "lat_at.npy")
time_sa = np.load(path_data_obs + "time_at.npy")
ssh_true = np.load(path_data_obs + "ssh_true.npy")
time_true = np.load(path_data_obs + "time_true.npy")

obs_dens_spatial = get_obs_spatial_density(lon_sa,lat_sa,lon_OCCIPUT.flatten(),lat_OCCIPUT.flatten())
obs_dens_spatial = obs_dens_spatial.reshape((41,41))


path_for_savefigs = cwd + '/figs/'

max_mode = 100
pc_true = (ssh_true - np.repeat(ssh_mean[np.newaxis],365,0)).dot(U[:,0:max_mode])
pc_AnDA = (ssh_AnDA - np.repeat(ssh_mean[np.newaxis],365,0)).dot(U[:,0:max_mode])
pc_OI = (ssh_OI - np.repeat(ssh_mean[np.newaxis],365,0)).dot(U[:,0:max_mode])
pc_COA = (ssh_COA - np.repeat(ssh_mean[np.newaxis],365,0)).dot(U[:,0:max_mode])

PDS_true = np.log(np.mean(pc_true**2,0))
PDS_AnDA = np.log(np.mean(pc_AnDA**2,0))
PDS_OI = np.log(np.mean(pc_OI**2,0))
PDS_COA = np.log(np.mean(pc_COA**2,0))

plt.plot(np.arange(PDS_true.shape[0]),PDS_true,'k')
plt.plot(np.arange(PDS_AnDA.shape[0]),PDS_AnDA,'r')
plt.plot(np.arange(PDS_OI.shape[0]),PDS_OI,'g')
plt.plot(np.arange(PDS_COA.shape[0]),PDS_COA,'g')
plt.legend(['True','AnDA','OI','COA'],fontsize=15)
plt.clf()

fig = plt.figure(figsize=(20,10))
ax2 = fig.add_subplot(111)
m2 = Basemap(llcrnrlon=-95.,llcrnrlat=17.31,urcrnrlon=-75.,urcrnrlat=33.3,
            projection='lcc',lat_1=20.,lat_2=40.,lon_0=-60.,
            resolution ='h',area_thresh=1000.)
m2.drawcoastlines()
m2.drawcountries()
m2.drawmapboundary(fill_color='#99ffff')
m2.fillcontinents(color='#cc9966',lake_color='#99ffff')
m2.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
m2.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])


x, y = m2(lon_OCCIPUT, lat_OCCIPUT)
x_patch = x
y_patch = y
i_center = np.where(idx_my_region == (41*41-1)/2)[0][0]
i_Florida = np.where(idx_my_region == 41*25+32)[0][0]
i_Yucatan = np.where(idx_my_region == 41*7+10)[0][0]
i_W = np.where(idx_my_region == 41*20+5)[0][0]
i_loopcurrent = np.where(idx_my_region == 41*17+20)[0][0]

dx_dy,dx_dx = np.gradient(x)
dy_dy,dy_dx = np.gradient(y)

dx = np.median(dx_dx)
dy = np.median(dy_dy)

g=9.81 
omega=7.2921e-5 
phi=lat_OCCIPUT*np.pi/180 
f_c=2*omega*np.sin(phi)


abserr_OI = np.abs(ssh_true - ssh_OI)
abserr_COA = np.abs(ssh_true - ssh_COA)
abserr_AnDA = np.abs(ssh_true - ssh_AnDA)

u_true = np.zeros((365,41,41));v_true = np.zeros((365,41,41));ke_true = np.zeros((365,41,41));ssh_map_true = np.zeros((365,41,41))
u_AnDA = np.zeros((365,41,41));v_AnDA = np.zeros((365,41,41));ke_AnDA = np.zeros((365,41,41));ssh_map_AnDA = np.zeros((365,41,41));var_map_AnDA = np.zeros((365,41,41))
u_OI = np.zeros((365,41,41));v_OI = np.zeros((365,41,41));ke_OI = np.zeros((365,41,41));ssh_map_OI = np.zeros((365,41,41));var_map_OI = np.zeros((365,41,41))
u_COA = np.zeros((365,41,41));v_COA = np.zeros((365,41,41));ke_COA = np.zeros((365,41,41));ssh_map_COA = np.zeros((365,41,41));var_map_COA = np.zeros((365,41,41))


vort_map_true = np.zeros((365,41,41))
vort_map_AnDA = np.zeros((365,41,41))
vort_map_OI = np.zeros((365,41,41))
vort_map_COA = np.zeros((365,41,41))

abserr_map_OI = np.zeros((365,41*41))
abserr_map_OI[:,idx_my_region]=np.copy(abserr_OI)
abserr_map_OI = np.reshape(abserr_map_OI,(365,41,41))
abserr_map_OI = ma.masked_values(abserr_map_OI,0)
abserr_map_OI_c = np.copy(abserr_map_OI[:,15:31,0:26])

abserr_map_COA = np.zeros((365,41*41))
abserr_map_COA[:,idx_my_region]=np.copy(abserr_COA)
abserr_map_COA = np.reshape(abserr_map_COA,(365,41,41))
abserr_map_COA = ma.masked_values(abserr_map_COA,0)
abserr_map_COA_c = np.copy(abserr_map_COA[:,15:31,0:26])

abserr_map_AnDA = np.zeros((365,41*41))
abserr_map_AnDA[:,idx_my_region]=np.copy(abserr_AnDA)
abserr_map_AnDA = np.reshape(abserr_map_AnDA,(365,41,41))
abserr_map_AnDA = ma.masked_values(abserr_map_AnDA,0)
abserr_map_AnDA_c = np.copy(abserr_map_AnDA[:,15:31,0:26])

for i in range(365):
    ssh_map_true_tmp = np.zeros(41*41)
    ssh_map_true_tmp[idx_my_region] = np.copy(ssh_true[i])
    ssh_map_true_tmp = np.reshape(ssh_map_true_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_true_tmp = ma.masked_values(ssh_map_true_tmp,0)
    u_true_tmp,v_true_tmp = mi.compute_U_V_image(ssh_map_true_tmp)
    u_true_tmp = u_true_tmp*g/f_c/dx
    v_true_tmp = v_true_tmp*g/f_c/dy
  
    ux_tmp,uy_tmp = mi.compute_U_V_image(u_true_tmp)
    uy_tmp = uy_tmp*g/f_c/dy
    vx_tmp,vy_tmp = mi.compute_U_V_image(v_true_tmp)
    vx_tmp = vx_tmp*g/f_c/dx
    vort_true_tmp = vx_tmp - uy_tmp  
  
    ke_true_tmp = u_true_tmp**2+v_true_tmp**2
    idx_u = ma.MaskedArray.nonzero(u_true_tmp.flatten())[0]
    u_true_tmp = u_true_tmp.flatten()[idx_u]
    u_true_tmp = mask_map(u_true_tmp,idx_u,(41,41))
    v_true_tmp = v_true_tmp.flatten()[idx_u]
    v_true_tmp = mask_map(v_true_tmp,idx_u,(41,41))
    ke_true_tmp = ke_true_tmp.flatten()[idx_u]
    ke_true_tmp = mask_map(ke_true_tmp,idx_u,(41,41))

    idx_vort = ma.MaskedArray.nonzero(vort_true_tmp.flatten())[0]
    vort_true_tmp = vort_true_tmp.flatten()[idx_vort]
    vort_true_tmp = mask_map(vort_true_tmp,idx_vort,(41,41))

    

    ssh_map_AnDA_tmp = np.zeros(41*41)
    ssh_map_AnDA_tmp[idx_my_region] = np.copy(ssh_AnDA[i])
    ssh_map_AnDA_tmp = np.reshape(ssh_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_AnDA_tmp = ma.masked_values(ssh_map_AnDA_tmp,0)
    u_AnDA_tmp,v_AnDA_tmp = mi.compute_U_V_image(ssh_map_AnDA_tmp)
    u_AnDA_tmp = u_AnDA_tmp*g/f_c/dx
    v_AnDA_tmp = v_AnDA_tmp*g/f_c/dy
    ke_AnDA_tmp = u_AnDA_tmp**2+v_AnDA_tmp**2
    ux_tmp,uy_tmp = mi.compute_U_V_image(u_AnDA_tmp)
    uy_tmp = uy_tmp*g/f_c/dy
    vx_tmp,vy_tmp = mi.compute_U_V_image(v_AnDA_tmp)
    vx_tmp = vx_tmp*g/f_c/dx
    vort_AnDA_tmp = vx_tmp - uy_tmp  
    idx_u = ma.MaskedArray.nonzero(u_true_tmp.flatten())[0]
    u_AnDA_tmp = u_AnDA_tmp.flatten()[idx_u]
    u_AnDA_tmp = mask_map(u_AnDA_tmp,idx_u,(41,41))
    v_AnDA_tmp = v_AnDA_tmp.flatten()[idx_u]
    v_AnDA_tmp = mask_map(v_AnDA_tmp,idx_u,(41,41))
    ke_AnDA_tmp = ke_AnDA_tmp.flatten()[idx_u]
    ke_AnDA_tmp = mask_map(ke_AnDA_tmp,idx_u,(41,41))   
    idx_vort = ma.MaskedArray.nonzero(vort_AnDA_tmp.flatten())[0]
    vort_AnDA_tmp = vort_AnDA_tmp.flatten()[idx_vort]
    vort_AnDA_tmp = mask_map(vort_AnDA_tmp,idx_vort,(41,41))

    var_map_AnDA_tmp = np.zeros(41*41)
    var_map_AnDA_tmp[idx_my_region] = var_xs_AnDA[i]
    var_map_AnDA_tmp = np.reshape(var_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_AnDA_tmp = ma.masked_values(var_map_AnDA_tmp,0)


    ssh_map_OI_tmp = np.zeros(41*41)
    ssh_map_OI_tmp[idx_my_region] = np.copy(ssh_OI[i])
    ssh_map_OI_tmp = np.reshape(ssh_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_OI_tmp = ma.masked_values(ssh_map_OI_tmp,0)
    u_OI_tmp,v_OI_tmp = mi.compute_U_V_image(ssh_map_OI_tmp)
    u_OI_tmp = u_OI_tmp*g/f_c/dx
    v_OI_tmp = v_OI_tmp*g/f_c/dy
    ke_OI_tmp = u_OI_tmp**2+v_OI_tmp**2
    ux_tmp,uy_tmp = mi.compute_U_V_image(u_OI_tmp)
    uy_tmp = uy_tmp*g/f_c/dy
    vx_tmp,vy_tmp = mi.compute_U_V_image(v_OI_tmp)
    vx_tmp = vx_tmp*g/f_c/dx
    vort_OI_tmp = vx_tmp - uy_tmp 
    idx_u = ma.MaskedArray.nonzero(u_true_tmp.flatten())[0]
    u_OI_tmp = u_OI_tmp.flatten()[idx_u]
    u_OI_tmp = mask_map(u_OI_tmp,idx_u,(41,41))
    v_OI_tmp = v_OI_tmp.flatten()[idx_u]
    v_OI_tmp = mask_map(v_OI_tmp,idx_u,(41,41))
    ke_OI_tmp = ke_OI_tmp.flatten()[idx_u]
    ke_OI_tmp = mask_map(ke_OI_tmp,idx_u,(41,41))   
    idx_vort = ma.MaskedArray.nonzero(vort_OI_tmp.flatten())[0]
    vort_OI_tmp = vort_OI_tmp.flatten()[idx_vort]
    vort_OI_tmp = mask_map(vort_OI_tmp,idx_vort,(41,41))
    var_map_OI_tmp = np.zeros(41*41)
    var_map_OI_tmp[idx_my_region] = var_xs_OI[i]
    var_map_OI_tmp = np.reshape(var_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_OI_tmp = ma.masked_values(var_map_OI_tmp,0)


    ssh_map_COA_tmp = np.zeros(41*41)
    ssh_map_COA_tmp[idx_my_region] = np.copy(ssh_COA[i])
    ssh_map_COA_tmp = np.reshape(ssh_map_COA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_COA_tmp = ma.masked_values(ssh_map_COA_tmp,0)
    u_COA_tmp,v_COA_tmp = mi.compute_U_V_image(ssh_map_COA_tmp)
    u_COA_tmp = u_COA_tmp*g/f_c/dx
    v_COA_tmp = v_COA_tmp*g/f_c/dy
    ke_COA_tmp = u_COA_tmp**2+v_COA_tmp**2
    ux_tmp,uy_tmp = mi.compute_U_V_image(u_COA_tmp)
    uy_tmp = uy_tmp*g/f_c/dy
    vx_tmp,vy_tmp = mi.compute_U_V_image(v_COA_tmp)
    vx_tmp = vx_tmp*g/f_c/dx
    vort_COA_tmp = vx_tmp - uy_tmp 
    idx_u = ma.MaskedArray.nonzero(u_true_tmp.flatten())[0]
    u_COA_tmp = u_COA_tmp.flatten()[idx_u]
    u_COA_tmp = mask_map(u_COA_tmp,idx_u,(41,41))
    v_COA_tmp = v_COA_tmp.flatten()[idx_u]
    v_COA_tmp = mask_map(v_COA_tmp,idx_u,(41,41))
    ke_COA_tmp = ke_COA_tmp.flatten()[idx_u]
    ke_COA_tmp = mask_map(ke_COA_tmp,idx_u,(41,41))   
    idx_vort = ma.MaskedArray.nonzero(vort_COA_tmp.flatten())[0]
    vort_COA_tmp = vort_COA_tmp.flatten()[idx_vort]
    vort_COA_tmp = mask_map(vort_COA_tmp,idx_vort,(41,41))
    var_map_COA_tmp = np.zeros(41*41)
    var_map_COA_tmp[idx_my_region] = var_xs_COA[i]
    var_map_COA_tmp = np.reshape(var_map_COA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_COA_tmp = ma.masked_values(var_map_COA_tmp,0)




    u_true[i] = np.copy(u_true_tmp);v_true[i] = np.copy(v_true_tmp);ke_true[i,:,:] = ke_true_tmp;ssh_map_true[i] = np.copy(ssh_map_true_tmp);vort_map_true[i] = np.copy(vort_true_tmp)
    u_AnDA[i] = np.copy(u_AnDA_tmp);v_AnDA[i] = np.copy(v_AnDA_tmp);ke_AnDA[i] = np.copy(ke_AnDA_tmp);ssh_map_AnDA[i] = np.copy(ssh_map_AnDA_tmp);var_map_AnDA[i] = np.copy(var_map_AnDA_tmp);vort_map_AnDA[i] = np.copy(vort_AnDA_tmp)
    u_OI[i] = np.copy(u_OI_tmp);v_OI[i] = np.copy(v_OI_tmp);ke_OI[i] = np.copy(ke_OI_tmp);ssh_map_OI[i] = np.copy(ssh_map_OI_tmp);var_map_OI[i] = np.copy(var_map_OI_tmp);vort_map_OI[i] = np.copy(vort_OI_tmp)
    u_COA[i] = np.copy(u_COA_tmp);v_COA[i] = np.copy(v_COA_tmp);ke_COA[i] = np.copy(ke_COA_tmp);ssh_map_COA[i] = np.copy(ssh_map_COA_tmp);var_map_COA[i] = np.copy(var_map_COA_tmp);vort_map_COA[i] = np.copy(vort_COA_tmp)

    if np.min(ke_true_tmp)<0:
	stop
    if np.min(ke_true)<0:
	stop


tmp = np.zeros((41,41))
tmp[15:31,0:26] = 1.
tmp =tmp.flatten()
tmp[idx_my_region] = tmp[idx_my_region] + 1.
idx_coast = np.where(tmp == 1.)[0]


u_true_c = u_true[:,15:31,0:26]
v_true_c = v_true[:,15:31,0:26]
ke_true_c = ke_true[:,15:31,0:26]
vort_true_c = vort_map_true[:,15:31,0:26]
u_AnDA_c = u_AnDA[:,15:31,0:26]
v_AnDA_c = v_AnDA[:,15:31,0:26]
ke_AnDA_c = ke_AnDA[:,15:31,0:26]
vort_AnDA_c = vort_map_AnDA[:,15:31,0:26]
u_OI_c = u_OI[:,15:31,0:26]
v_OI_c = v_OI[:,15:31,0:26]
ke_OI_c = ke_OI[:,15:31,0:26]
vort_OI_c = vort_map_OI[:,15:31,0:26]
u_COA_c = u_COA[:,15:31,0:26]
v_COA_c = v_COA[:,15:31,0:26]
ke_COA_c = ke_COA[:,15:31,0:26]
vort_COA_c = vort_map_COA[:,15:31,0:26]

u_true_F = u_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
v_true_F = v_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ke_true_F = ke_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
vort_true_F = vort_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
u_AnDA_F = u_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
v_AnDA_F = v_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ke_AnDA_F = ke_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
vort_AnDA_F = vort_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
u_OI_F = u_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
v_OI_F = v_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ke_OI_F = ke_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
vort_OI_F = vort_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
u_COA_F = u_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
v_COA_F = v_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ke_COA_F = ke_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
vort_COA_F = vort_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]


u_true_Y = u_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
v_true_Y = v_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ke_true_Y = ke_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
vort_true_Y = vort_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
u_AnDA_Y = u_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
v_AnDA_Y = v_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ke_AnDA_Y = ke_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
vort_AnDA_Y = vort_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
u_OI_Y = u_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
v_OI_Y = v_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ke_OI_Y = ke_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
vort_OI_Y = vort_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
u_COA_Y = u_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
v_COA_Y = v_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ke_COA_Y = ke_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
vort_COA_Y = vort_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]



u_true_co = u_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
v_true_co = v_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ke_true_co = ke_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
vort_true_co = vort_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
u_AnDA_co = u_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
v_AnDA_co = v_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ke_AnDA_co = ke_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
vort_AnDA_co = vort_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
u_OI_co = u_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
v_OI_co = v_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ke_OI_co = ke_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
vort_OI_co = vort_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
u_COA_co = u_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
v_COA_co = v_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ke_COA_co = ke_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
vort_COA_co = vort_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]


maxvort_map_true = np.nanmax(vort_map_true,axis = 0)
maxvort_map_AnDA = np.nanmax(vort_map_AnDA,axis = 0)
maxvort_map_OI = np.nanmax(vort_map_OI,axis = 0)
maxvort_map_COA = np.nanmax(vort_map_COA,axis = 0)

minvort_map_true = np.nanmin(vort_map_true,axis = 0)
minvort_map_AnDA = np.nanmin(vort_map_AnDA,axis = 0)
minvort_map_OI = np.nanmin(vort_map_OI,axis = 0)
minvort_map_COA = np.nanmin(vort_map_COA,axis = 0)

maxvort_map_true = ma.masked_values(maxvort_map_true,0)
maxvort_map_AnDA = ma.masked_values(maxvort_map_AnDA,0)
maxvort_map_OI = ma.masked_values(maxvort_map_OI,0)
maxvort_map_COA = ma.masked_values(maxvort_map_COA,0)
minvort_map_true = ma.masked_values(minvort_map_true,0)
minvort_map_AnDA = ma.masked_values(minvort_map_AnDA,0)
minvort_map_OI = ma.masked_values(minvort_map_OI,0)
minvort_map_COA = ma.masked_values(minvort_map_COA,0)


ssh_map_true_c = ssh_map_true[:,15:31,0:26]
ssh_map_AnDA_c = ssh_map_AnDA[:,15:31,0:26]
ssh_map_OI_c = ssh_map_OI[:,15:31,0:26]
ssh_map_COA_c = ssh_map_COA[:,15:31,0:26]
var_map_AnDA_c = var_map_AnDA[:,15:31,0:26]
var_map_OI_c = var_map_OI[:,15:31,0:26]
var_map_COA_c = var_map_COA[:,15:31,0:26]

ssh_true_F = ssh_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ssh_AnDA_F = ssh_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ssh_OI_F = ssh_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
ssh_COA_F = ssh_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
var_AnDA_F = var_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
var_OI_F = var_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]
var_COA_F = var_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_FLcoast]

ssh_true_Y = ssh_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ssh_AnDA_Y = ssh_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ssh_OI_Y = ssh_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
ssh_COA_Y = ssh_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
var_AnDA_Y = var_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
var_OI_Y = var_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]
var_COA_Y = var_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_YCcoast]

ssh_true_co = ssh_map_true.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ssh_AnDA_co = ssh_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ssh_OI_co = ssh_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
ssh_COA_co = ssh_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
var_AnDA_co = var_map_AnDA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
var_OI_co = var_map_OI.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]
var_COA_co = var_map_COA.reshape((ssh_map_true.shape[0],ssh_map_true.shape[1]*ssh_map_true.shape[2]))[:,idx_coast]


#print "Whole region: ",'RMSE-ssh(AnDA,OI,COA):',np.sqrt(np.mean((ssh_map_true-ssh_map_AnDA).reshape((365,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((ssh_map_true-ssh_map_OI).reshape((365,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((ssh_map_true-ssh_map_COA).reshape((365,41*41))[:,idx_my_region]**2))
#print "Whole region: ", "RMSE-uv(AnDA,OI,COA):",np.sqrt(np.mean((u_true-u_AnDA).reshape((365,41*41))[:,idx_my_region]**2+(v_true-v_AnDA).reshape((365,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((u_true-u_OI).reshape((365,41*41))[:,idx_my_region]**2+(v_true-v_OI).reshape((365,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((u_true-u_COA).reshape((365,41*41))[:,idx_my_region]**2+(v_true-v_COA).reshape((365,41*41))[:,idx_my_region]**2))
#print "Central region: ",'RMSE-ssh(AnDA,OI,COA):',np.sqrt(np.mean((ssh_map_true_c-ssh_map_AnDA_c)**2)),np.sqrt(np.mean((ssh_map_true_c-ssh_map_OI_c)**2)),np.sqrt(np.mean((ssh_map_true_c-ssh_map_COA_c)**2))
#print "Central region: ", "RMSE-uv(AnDA,OI,COA):",np.sqrt(np.mean((u_true_c-u_AnDA_c)**2+(v_true_c-v_AnDA_c)**2)),np.sqrt(np.mean((u_true_c-u_OI_c)**2+(v_true_c-v_OI_c)**2)),np.sqrt(np.mean((u_true_c-u_COA_c)**2+(v_true_c-v_COA_c)**2))



print '========================RMSE cutoff the head and tail========================='
print "Whole region: ",'RMSE-ssh(AnDA,OI,COA):',np.sqrt(np.mean((ssh_map_true[d:365-d]-ssh_map_AnDA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((ssh_map_true[d:365-d]-ssh_map_OI[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((ssh_map_true[d:365-d]-ssh_map_COA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2))
print "Whole region: ", "RMSE-uv(AnDA,OI,COA):",np.sqrt(np.mean((u_true[d:365-d]-u_AnDA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2+(v_true[d:365-d]-v_AnDA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((u_true[d:365-d]-u_OI[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2+(v_true[d:365-d]-v_OI[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((u_true[d:365-d]-u_COA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2+(v_true[d:365-d]-v_COA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2))
print "Whole region: ",'RMSE-vort(AnDA,OI,COA):',np.sqrt(np.mean((vort_map_true[d:365-d]-vort_map_AnDA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((vort_map_true[d:365-d]-vort_map_OI[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2)),np.sqrt(np.mean((vort_map_true[d:365-d]-vort_map_COA[d:365-d]).reshape((365-2*d,41*41))[:,idx_my_region]**2))
print "----------------------------------------------------------------"
print "Central region: ",'RMSE-ssh(AnDA,OI,COA):',np.sqrt(np.mean((ssh_map_true_c[d:365-d]-ssh_map_AnDA_c[d:365-d])**2)),np.sqrt(np.mean((ssh_map_true_c[d:365-d]-ssh_map_OI_c[d:365-d])**2)),np.sqrt(np.mean((ssh_map_true_c[d:365-d]-ssh_map_COA_c[d:365-d])**2))
print "Central region: ", "RMSE-uv(AnDA,OI,COA):",np.sqrt(np.mean((u_true_c[d:365-d]-u_AnDA_c[d:365-d])**2+(v_true_c[d:365-d]-v_AnDA_c[d:365-d])**2)),np.sqrt(np.mean((u_true_c[d:365-d]-u_OI_c[d:365-d])**2+(v_true_c[d:365-d]-v_OI_c[d:365-d])**2)),np.sqrt(np.mean((u_true_c[d:365-d]-u_COA_c[d:365-d])**2+(v_true_c[d:365-d]-v_COA_c[d:365-d])**2))
print "Central region: ",'RMSE-vort(AnDA,OI,COA):',np.sqrt(np.mean((vort_true_c[d:365-d]-vort_AnDA_c[d:365-d])**2)),np.sqrt(np.mean((vort_true_c[d:365-d]-vort_OI_c[d:365-d])**2)),np.sqrt(np.mean((vort_true_c[d:365-d]-vort_COA_c[d:365-d])**2))

print "----------------------------------------------------------------"
RMSE_ssh_F = np.array([np.sqrt(np.mean((ssh_true_F[d:365-d] - ssh_AnDA_F[d:365-d])**2)),np.sqrt(np.mean((ssh_true_F[d:365-d] - ssh_OI_F[d:365-d])**2)),np.sqrt(np.mean((ssh_true_F[d:365-d] - ssh_COA_F[d:365-d])**2))])
RMSE_uv_F = np.array([np.sqrt(np.mean((u_true_F[d:365-d] - u_AnDA_F[d:365-d])**2 + (v_true_F[d:365-d] - v_AnDA_F[d:365-d])**2)),np.sqrt(np.mean((u_true_F[d:365-d] - u_OI_F[d:365-d])**2 + (v_true_F[d:365-d] - v_OI_F[d:365-d])**2)),np.sqrt(np.mean((u_true_F[d:365-d] - u_COA_F[d:365-d])**2 + (v_true_F[d:365-d] - v_COA_F[d:365-d])**2))])
RMSE_vort_F = np.array([np.sqrt(np.mean((vort_true_F[d:365-d] - vort_AnDA_F[d:365-d])**2)),np.sqrt(np.mean((vort_true_F[d:365-d] - vort_OI_F[d:365-d])**2)),np.sqrt(np.mean((vort_true_F[d:365-d] - vort_COA_F[d:365-d])**2))])
print "Florida Coast: ", "RMSE-ssh(AnDA,OI,COA): ", RMSE_ssh_F[0],RMSE_ssh_F[1],RMSE_ssh_F[2]
print "Florida Coast: ", "RMSE-uv(AnDA,OI,COA): ", RMSE_uv_F[0],RMSE_uv_F[1],RMSE_uv_F[2]
print "Florida Coast: ", "RMSE-vort(AnDA,OI,COA): ",RMSE_vort_F[0],RMSE_vort_F[1],RMSE_vort_F[2]


print "----------------------------------------------------------------"
RMSE_ssh_Y = np.array([np.sqrt(np.mean((ssh_true_Y[d:365-d] - ssh_AnDA_Y[d:365-d])**2)),np.sqrt(np.mean((ssh_true_Y[d:365-d] - ssh_OI_Y[d:365-d])**2)),np.sqrt(np.mean((ssh_true_Y[d:365-d] - ssh_COA_Y[d:365-d])**2))])
RMSE_uv_Y = np.array([np.sqrt(np.mean((u_true_Y[d:365-d] - u_AnDA_Y[d:365-d])**2 + (v_true_Y[d:365-d] - v_AnDA_Y[d:365-d])**2)),np.sqrt(np.mean((u_true_Y[d:365-d] - u_OI_Y[d:365-d])**2 + (v_true_Y[d:365-d] - v_OI_Y[d:365-d])**2)),np.sqrt(np.mean((u_true_Y[d:365-d] - u_COA_Y[d:365-d])**2 + (v_true_Y[d:365-d] - v_COA_Y[d:365-d])**2))])
RMSE_vort_Y = np.array([np.sqrt(np.mean((vort_true_Y[d:365-d] - vort_AnDA_Y[d:365-d])**2)),np.sqrt(np.mean((vort_true_Y[d:365-d] - vort_OI_Y[d:365-d])**2)),np.sqrt(np.mean((vort_true_Y[d:365-d] - vort_COA_Y[d:365-d])**2))])
print "Yucatan Coast: ", "RMSE-ssh(AnDA,OI,COA): ", RMSE_ssh_Y[0],RMSE_ssh_Y[1],RMSE_ssh_Y[2]
print "Yucatan Coast: ", "RMSE-uv(AnDA,OI,COA): ", RMSE_uv_Y[0],RMSE_uv_Y[1],RMSE_uv_Y[2]
print "Yucatan Coast: ", "RMSE-vort(AnDA,OI,COA): ", RMSE_vort_Y[0],RMSE_vort_Y[1],RMSE_vort_Y[2]

print "----------------------------------------------------------------"
RMSE_ssh_YF = np.sqrt((RMSE_ssh_F**2.*60 + RMSE_ssh_Y**2.*33.)/93.)
RMSE_uv_YF = np.sqrt((RMSE_uv_F**2.*60 + RMSE_uv_Y**2.*33.)/93.)
RMSE_vort_YF = np.sqrt((RMSE_vort_F**2.*60 + RMSE_vort_Y**2.*33.)/93.)
print "Florida+Yucatan Coast: ", "RMSE-ssh(AnDA,OI,COA): ", RMSE_ssh_YF[0],RMSE_ssh_YF[1],RMSE_ssh_YF[2]
print "Florida+Yucatan Coast: ", "RMSE-uv(AnDA,OI,COA): ", RMSE_uv_YF[0],RMSE_uv_YF[1],RMSE_uv_YF[2]
print "Florida+Yucatan Coast: ", "RMSE-vort(AnDA,OI,COA): ",RMSE_vort_YF[0],RMSE_vort_YF[1],RMSE_vort_YF[2]



print "----------------------------------------------------------------"
print "All Coasts: ", "RMSE-ssh(AnDA,OI,COA): ",np.sqrt(np.mean((ssh_true_co[d:365-d] - ssh_AnDA_co[d:365-d])**2)),np.sqrt(np.mean((ssh_true_co[d:365-d] - ssh_OI_co[d:365-d])**2)),np.sqrt(np.mean((ssh_true_co[d:365-d] - ssh_COA_co[d:365-d])**2))
print "All Coasts: ", "RMSE-uv(AnDA,OI,COA): ",np.sqrt(np.mean((u_true_co[d:365-d] - u_AnDA_co[d:365-d])**2 + (v_true_co[d:365-d] - v_AnDA_co[d:365-d])**2)),np.sqrt(np.mean((u_true_co[d:365-d] - u_OI_co[d:365-d])**2 + (v_true_co[d:365-d] - v_OI_co[d:365-d])**2)),np.sqrt(np.mean((u_true_co[d:365-d] - u_COA_co[d:365-d])**2 + (v_true_co[d:365-d] - v_COA_co[d:365-d])**2))
print "All Coasts: ", "RMSE-vort(AnDA,OI,COA): ",np.sqrt(np.mean((vort_true_co[d:365-d] - vort_AnDA_co[d:365-d])**2)),np.sqrt(np.mean((vort_true_co[d:365-d] - vort_OI_co[d:365-d])**2)),np.sqrt(np.mean((vort_true_co[d:365-d] - vort_COA_co[d:365-d])**2))

first_days_per_month = np.array([0,31,60,91,121,152,182,213,244,274,305,335])
names_per_month = ('Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sep.','Oct.','Nov.','Dec.')



cbar_ax1 = fig.add_axes([0.15, 0.15, 0.01, 0.4])
cbar_ax2 = fig.add_axes([0.85, 0.15, 0.01, 0.7])
lats_box = [ np.min(lat_OCCIPUT[15:31,0:26]), np.max(lat_OCCIPUT[15:31,0:26]), np.max(lat_OCCIPUT[15:31,0:26]), np.min(lat_OCCIPUT[15:31,0:26]) ]
lons_box = [ np.min(lon_OCCIPUT[15:31,0:26]), np.min(lon_OCCIPUT[15:31,0:26]), np.max(lon_OCCIPUT[15:31,0:26]), np.max(lon_OCCIPUT[15:31,0:26]) ]


fig2 = plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(311)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_center],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_center],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_center],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_center],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')
plt.title( 'Reconstruction results at ($85^{\circ}W, 25^{\circ}N$) in 2004 ',fontsize = 20)

fig2.add_subplot(312)
plt.plot(np.arange(d,365-d), np.sqrt(ke_true[d:365-d,20,20]),'r',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_AnDA[d:365-d,20,20]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_OI[d:365-d,20,20]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_COA[d:365-d,20,20]),'g',lw=2)
plt.xlim(0,365)
plt.ylabel('velocity(m/s)',fontsize = 20)
plt.yticks(fontsize=20)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')

fig2.add_subplot(313)
plt.plot(np.arange(d,365-d), vort_map_true[d:365-d,20,20],'r',lw=2)
plt.plot(np.arange(d,365-d), vort_map_AnDA[d:365-d,20,20],'k',lw=2)
plt.plot(np.arange(d,365-d), vort_map_OI[d:365-d,20,20],'b',lw=2)
plt.plot(np.arange(d,365-d), vort_map_COA[d:365-d,20,20],'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('vorticity',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')

plt.savefig(path_for_savefigs + 'TimeSeries.png',bbox_inches = 'tight')
plt.clf()


fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_center],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_center],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_center],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_center],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at ($85^{\circ}W, 25^{\circ}N$) in 2004 ',fontsize = 20)
plt.title( 'Reconstruction results at the loop current in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries1.png',bbox_inches = 'tight')
plt.clf()

fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), vort_map_true[d:365-d,20,20],'r',lw=2)
plt.plot(np.arange(d,365-d), vort_map_AnDA[d:365-d,20,20],'k',lw=2)
plt.plot(np.arange(d,365-d), vort_map_OI[d:365-d,20,20],'b',lw=2)
plt.plot(np.arange(d,365-d), vort_map_COA[d:365-d,20,20],'b',lw=2)
plt.xlim(0,365)
plt.ylabel('vorticity',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at ($85^{\circ}W, 25^{\circ}N$) in 2004 ',fontsize = 20)
plt.title( 'Reconstruction results at the loop current in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries_vort.png',bbox_inches = 'tight')
plt.clf()



fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(ke_true[d:365-d,20,20]),'r',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_AnDA[d:365-d,20,20]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_OI[d:365-d,20,20]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_COA[d:365-d,20,20]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('velocity(m/s)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at ($85^{\circ}W, 25^{\circ}N$) in 2004 ',fontsize = 20)
plt.title( 'Reconstruction results at the loop current in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries2.png',bbox_inches = 'tight')
plt.clf()





fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_Florida],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_Florida],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_Florida],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_Florida],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results near Florida in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries1_F.png',bbox_inches = 'tight')
plt.clf()



fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_center],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_center],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_center],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_center],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

labels = [item.get_text() for item in ax1.get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax1.set_xticklabels(empty_string_labels)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at ($85^{\circ}W, 25^{\circ}N$) in 2004 ',fontsize = 20)
plt.title( 'Reconstruction results at the loop current in 2004 ',fontsize = 20)

ax2=fig2.add_subplot(212)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_Florida],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_Florida],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_Florida],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_Florida],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results near Florida in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries1_LF.png',bbox_inches = 'tight')
plt.clf()



fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(ke_true[d:365-d,25,32]),'r',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_AnDA[d:365-d,25,32]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_OI[d:365-d,25,32]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_COA[d:365-d,25,32]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('velocity(m/s)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results near Florida in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries2_F.png',bbox_inches = 'tight')
plt.clf()


fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_Yucatan],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_Yucatan],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_Yucatan],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_Yucatan],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results near Yucatan in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries1_Y.png',bbox_inches = 'tight')
plt.clf()



fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(ke_true[d:365-d,7,10]),'r',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_AnDA[d:365-d,7,10]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_OI[d:365-d,7,10]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_COA[d:365-d,7,10]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('velocity(m/s)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results near Yucatan in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries2_Y.png',bbox_inches = 'tight')
plt.clf()




fig2=plt.figure(figsize=(16,9))
ax1=fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), ssh_true[d:365-d,i_W],'r',lw=2)
plt.plot(np.arange(d,365-d), ssh_AnDA[d:365-d,i_W],'k',lw=2)
plt.plot(np.arange(d,365-d), ssh_OI[d:365-d,i_W],'b',lw=2)
plt.plot(np.arange(d,365-d), ssh_COA[d:365-d,i_W],'g',lw=2)
plt.xlim(0,365)
plt.ylabel('SSH(m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)

plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at the GoM center in 2004 ',fontsize = 20)

plt.savefig(path_for_savefigs + 'TimeSeries1_W.png',bbox_inches = 'tight')
plt.clf()



fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(ke_true[d:365-d,20,5]),'r',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_AnDA[d:365-d,20,5]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_OI[d:365-d,20,5]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(ke_COA[d:365-d,20,5]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('velocity(m/s)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['Truth','AnDA','OI','OI$_{COA}$'],fontsize=20,loc='lower right')
plt.title( 'Reconstruction results at the GoM center in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries2_W.png',bbox_inches = 'tight')
plt.clf()


fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.abs(ssh_true[d:365-d,i_loopcurrent]-ssh_AnDA[d:365-d,i_loopcurrent]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.abs(ssh_true[d:365-d,i_loopcurrent]-ssh_OI[d:365-d,i_loopcurrent]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.abs(ssh_true[d:365-d,i_loopcurrent]-ssh_COA[d:365-d,i_loopcurrent]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('Error (m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')
plt.title( 'Reconstruction error of SSH at loop current in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries3_E.png',bbox_inches = 'tight')
plt.clf()

fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(var_xs_AnDA[d:365-d,i_loopcurrent]),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(var_xs_OI[d:365-d,i_loopcurrent]),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(var_xs_COA[d:365-d,i_loopcurrent]),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('std. dev. (m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')
plt.title( 'Reconstruction std. dev. of SSH at loop current in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries3_Ps.png',bbox_inches = 'tight')
plt.clf()


fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.mean(np.mean(np.abs(ssh_map_true_c[d:365-d,:]-ssh_map_AnDA_c[d:365-d,:]),1),1),'k',lw=2)
plt.plot(np.arange(d,365-d), np.mean(np.mean(np.abs(ssh_map_true_c[d:365-d,:]-ssh_map_OI_c[d:365-d,:]),1),1),'b',lw=2)
plt.plot(np.arange(d,365-d), np.mean(np.mean(np.abs(ssh_map_true_c[d:365-d,:]-ssh_map_COA_c[d:365-d,:]),1),1),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('Error (m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')
plt.title( 'Spatial mean reconstruction error of SSH in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries4_E.png',bbox_inches = 'tight')
plt.clf()

fig2=plt.figure(figsize=(16,9))
fig2.add_subplot(211)
plt.plot(np.arange(d,365-d), np.sqrt(np.mean(np.mean(var_map_AnDA_c[d:365-d,:,:],1),1)),'k',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(np.mean(np.mean(var_map_OI_c[d:365-d,:,:],1),1)),'b',lw=2)
plt.plot(np.arange(d,365-d), np.sqrt(np.mean(np.mean(var_map_COA_c[d:365-d,:,:],1),1)),'g',lw=2)
plt.xlim(0,365)
plt.xticks(first_days_per_month,names_per_month,fontsize=20)
plt.ylabel('std. dev. (m)',fontsize = 20)
plt.yticks(fontsize=20)
plt.legend(['AnDA','OI','OI$_{COA}$'],fontsize=20,loc='upper right')
plt.title( 'Spatial mean reconstruction std. dev. of SSH at loop current in 2004 ',fontsize = 20)
plt.savefig(path_for_savefigs + 'TimeSeries4_Ps.png',bbox_inches = 'tight')
plt.clf()



plt.close('all')
ssh_map_true_mean = np.zeros((41,41))
ssh_map_AnDA_mean = np.zeros((41,41))
ssh_map_OI_mean = np.zeros((41,41))
ssh_map_COA_mean = np.zeros((41,41))


RMSE_map_AnDA = np.zeros((41,41))
RMSE_map_OI = np.zeros((41,41))
RMSE_map_COA = np.zeros((41,41))
stdev_map_AnDA_mean = np.zeros((41,41))
stdev_map_OI_mean = np.zeros((41,41))
stdev_map_COA_mean = np.zeros((41,41))


for t in tqdm(time_true[d:365-d]):
    i_xs = (t - np.min(time_OI)).astype(int)
    i_sa = np.where((time_sa >= t-3) & (time_sa < t+4))[0]

    date_now = date.fromordinal(t.astype(int))
    year = date_now.year
    month = month_str(date_now.month)
    day = day_str(date_now.day)


    ssh_map_true_tmp = np.zeros(41*41)
    ssh_map_true_tmp[idx_my_region] = np.copy(ssh_true[i_xs])
    ssh_map_true_tmp = np.reshape(ssh_map_true_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_true_tmp = ma.masked_values(ssh_map_true_tmp,0)



    ssh_map_AnDA_tmp = np.zeros(41*41)
    ssh_map_AnDA_tmp[idx_my_region] = np.copy(ssh_AnDA[i_xs])
    ssh_map_AnDA_tmp = np.reshape(ssh_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_AnDA_tmp = ma.masked_values(ssh_map_AnDA_tmp,0)
    var_map_AnDA_tmp = np.zeros(41*41)
    var_map_AnDA_tmp[idx_my_region] = var_xs_AnDA[i_xs]
    var_map_AnDA_tmp = np.reshape(var_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_AnDA_tmp = ma.masked_values(var_map_AnDA_tmp,0)

    stdev_map_AnDA_mean = stdev_map_AnDA_mean+np.sqrt(var_map_AnDA_tmp)
    RMSE_map_AnDA = RMSE_map_AnDA + (ssh_map_true_tmp - ssh_map_AnDA_tmp)**2.

    ssh_map_OI_tmp = np.zeros(41*41)
    ssh_map_OI_tmp[idx_my_region] = np.copy(ssh_OI[i_xs])
    ssh_map_OI_tmp = np.reshape(ssh_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_OI_tmp = ma.masked_values(ssh_map_OI_tmp,0)
    var_map_OI_tmp = np.zeros(41*41)
    var_map_OI_tmp[idx_my_region] = var_xs_OI[i_xs]
    var_map_OI_tmp = np.reshape(var_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_OI_tmp = ma.masked_values(var_map_OI_tmp,0)

    stdev_map_OI_mean = stdev_map_OI_mean+np.sqrt(var_map_OI_tmp)
    RMSE_map_OI = RMSE_map_OI + (ssh_map_true_tmp - ssh_map_OI_tmp)**2.

    ssh_map_COA_tmp = np.zeros(41*41)
    ssh_map_COA_tmp[idx_my_region] = np.copy(ssh_COA[i_xs])
    ssh_map_COA_tmp = np.reshape(ssh_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_COA_tmp = ma.masked_values(ssh_map_COA_tmp,0)
    var_map_COA_tmp = np.zeros(41*41)
    var_map_COA_tmp[idx_my_region] = var_xs_COA[i_xs]
    var_map_COA_tmp = np.reshape(var_map_COA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_COA_tmp = ma.masked_values(var_map_COA_tmp,0)

    stdev_map_COA_mean = stdev_map_COA_mean+np.sqrt(var_map_COA_tmp)
    RMSE_map_COA = RMSE_map_COA + (ssh_map_true_tmp - ssh_map_COA_tmp)**2.

stdev_map_AnDA_mean = stdev_map_AnDA_mean/(365.-2*d)
stdev_map_OI_mean = stdev_map_OI_mean/(365.-2*d)
stdev_map_COA_mean = stdev_map_COA_mean/(365.-2*d)
RMSE_map_AnDA = np.sqrt(RMSE_map_AnDA/(365.-2*d))
RMSE_map_OI = np.sqrt(RMSE_map_OI/(365.-2*d))
RMSE_map_COA = np.sqrt(RMSE_map_COA/(365.-2*d))

fig4 = plt.figure(figsize=(20,14))
ax5=fig4.add_subplot(231)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,stdev_map_AnDA_mean,0.,0.016,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',0.,0.016,'YlOrRd')
plt.clim([0.,0.016])
cb=m.colorbar(location='bottom',ticks=[0.,0.016])
cb.ax.set_xticklabels(['0', '0.016'],fontsize = 20)  # vertically oriented colorbar
plt.title('AnDA (std)',fontsize = 30)


fig4.add_subplot(232)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,stdev_map_OI_mean,0.,0.03,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',0.,0.03,'YlOrRd')
plt.clim([0.,0.03])
cb=m.colorbar(location='bottom',ticks=[0.,0.03])
cb.ax.set_xticklabels(['0','0.03'],fontsize = 20)  # vertically oriented colorbar
plt.title('OI (std)',fontsize = 30)

fig4.add_subplot(233)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.sqrt(varclim_map),0.,0.3,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',0,0.3,'YlOrRd')
plt.clim([0.,0.3])
cb=m.colorbar(location='bottom',ticks=[0., 0.3])
cb.ax.set_xticklabels(['0','0.3'],fontsize = 20)  # vertically oriented colorbar
plt.title('Background std',fontsize = 30)

ax5=fig4.add_subplot(234)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,RMSE_map_AnDA,0.,0.03,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
plt.clim([0.,0.03])
cb=m.colorbar(location='bottom',ticks=[0.,0.03])
cb.ax.set_xticklabels(['0','0.03'],fontsize = 20)  # vertically oriented colorbar
plt.title('AnDA (error)',fontsize = 30)


fig4.add_subplot(235)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,RMSE_map_OI,0.,0.03,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
plt.clim([0.,0.03])
cb=m.colorbar(location='bottom',ticks=[0.,0.03])
cb.ax.set_xticklabels(['0','0.03'],fontsize = 20)  # vertically oriented colorbar
plt.title('OI (error)',fontsize = 30)

fig4.add_subplot(236)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,obs_dens_spatial,0.,270.,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
plt.clim([0.,270.])
cb=m.colorbar(location='bottom',ticks=[0., 270.])
cb.ax.set_xticklabels(['0','max'],fontsize = 20)  # vertically oriented colorbar
plt.title('Obs frequency',fontsize = 30)

plt.savefig(path_for_savefigs + 'stdev_RMSE' + '.png',bbox_inches = 'tight')
plt.close('all')


fig4 = plt.figure(figsize=(20,20))
fig4.add_subplot(121)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.sqrt(varclim_map),0.,0.3,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',0,0.3,'YlOrRd')
plt.clim([0.,0.3])
cb=m.colorbar(location='bottom',ticks=[0., 0.3])
cb.ax.set_xticklabels(['0','0.3'],fontsize = 20)  # vertically oriented colorbar

fig4.add_subplot(122)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,obs_dens_spatial,0.,270.,'YlOrRd')
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
plt.clim([0.,270.])
cb=m.colorbar(location='bottom',ticks=[0., 270.])
cb.ax.set_xticklabels(['0','max'],fontsize = 20)  # vertically oriented colorbar

plt.savefig(path_for_savefigs + 'obsdens_varclim' + '.png',bbox_inches = 'tight')
plt.close('all')


fig4 = plt.figure(figsize=(20,20))
ax5=fig4.add_subplot(231)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,maxvort_map_true,0.,0.13,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([0.,0.13])
cb=m.colorbar(location='bottom',ticks=[0.,0.04,0.08,0.13])
cb.ax.set_xticklabels(['0', '0.04','0.08','0.13'],fontsize = 20)  # vertically oriented colorbar
plt.ylabel('max vorticity',fontsize = 30)
plt.title('Truth',fontsize = 30)

fig4.add_subplot(232)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,maxvort_map_AnDA,0.,0.13,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([0.,0.13])
cb=m.colorbar(location='bottom',ticks=[0.,0.04,0.08,0.13])
cb.ax.set_xticklabels(['0', '0.04','0.08','0.13'],fontsize = 20)  # vertically oriented colorbar
plt.title('AnDA',fontsize = 30)

fig4.add_subplot(233)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,maxvort_map_OI,0.,0.13,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([0.,0.13])
cb=m.colorbar(location='bottom',ticks=[0.,0.04,0.08,0.13])
cb.ax.set_xticklabels(['0', '0.04','0.08','0.13'],fontsize = 20)  # vertically oriented colorbar
plt.title('OI',fontsize = 30)

ax5=fig4.add_subplot(234)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,minvort_map_true,-0.08,0.,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([-0.08,0.])
cb=m.colorbar(location='bottom',ticks=[-0.08,-0.05,-0.02,0.])
cb.ax.set_xticklabels(['-0.08', '-0.05','-0.02','0.'],fontsize = 20)  # vertically oriented colorbar
plt.ylabel('min vorticity',fontsize = 30)
plt.title('Truth',fontsize = 30)

fig4.add_subplot(235)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,minvort_map_AnDA,-0.08,0.,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([-0.08,0.])
cb=m.colorbar(location='bottom',ticks=[-0.08,-0.05,-0.02,0.])
cb.ax.set_xticklabels(['-0.08', '-0.05','-0.02','0.'],fontsize = 20)  # vertically oriented colorbar
plt.title('AnDA',fontsize = 30)

fig4.add_subplot(236)
m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,minvort_map_OI,-0.08,0.,mycmap)
mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
plt.clim([-0.08,0.])
cb=m.colorbar(location='bottom',ticks=[-0.08,-0.05,-0.02,0.])
cb.ax.set_xticklabels(['-0.08', '-0.05','-0.02','0.'],fontsize = 20)  # vertically oriented colorbar
plt.title('OI',fontsize = 30)


plt.savefig(path_for_savefigs + 'maxminvort' + '.png',bbox_inches = 'tight')
plt.close('all')


for t in tqdm(time_true[np.array([68,251])]):
#for t in tqdm(time_true[np.arange(365)]):
    if t < np.min(time_OI)-290:
	continue
    i_xs = (t - np.min(time_OI)).astype(int)
    i_sa = np.where((time_sa >= t-3) & (time_sa < t+4))[0]

    date_now = date.fromordinal(t.astype(int))
    year = date_now.year
    month = month_str(date_now.month)
    day = day_str(date_now.day)

    d_mean = 15
    vmax_err_OI = 0.03
    vmax_err_AnDA = 0.03
    vmax_err_COA = 0.03
    vmax_stdev_OI = 0.03
    vmax_stdev_AnDA = 0.03
    vmax_stdev_COA = 0.03

    ssh_map_true_tmp = np.zeros(41*41)
    ssh_map_true_tmp[idx_my_region] = np.copy(ssh_true[i_xs])
    ssh_map_true_tmp = np.reshape(ssh_map_true_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_true_tmp = ma.masked_values(ssh_map_true_tmp,0)


    ssh_map_AnDA_tmp = np.zeros(41*41)
    ssh_map_AnDA_tmp[idx_my_region] = np.copy(ssh_AnDA[i_xs])
    ssh_map_AnDA_tmp = np.reshape(ssh_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_AnDA_tmp = ma.masked_values(ssh_map_AnDA_tmp,0)
    var_map_AnDA_tmp = np.zeros(41*41)
    var_map_AnDA_tmp[idx_my_region] = var_xs_AnDA[i_xs]
    var_map_AnDA_tmp = np.reshape(var_map_AnDA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_AnDA_tmp = ma.masked_values(var_map_AnDA_tmp,0)

    ssh_map_OI_tmp = np.zeros(41*41)
    ssh_map_OI_tmp[idx_my_region] = np.copy(ssh_OI[i_xs])
    ssh_map_OI_tmp = np.reshape(ssh_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_OI_tmp = ma.masked_values(ssh_map_OI_tmp,0)
    var_map_OI_tmp = np.zeros(41*41)
    var_map_OI_tmp[idx_my_region] = var_xs_OI[i_xs]
    var_map_OI_tmp = np.reshape(var_map_OI_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_OI_tmp = ma.masked_values(var_map_OI_tmp,0)

    ssh_map_COA_tmp = np.zeros(41*41)
    ssh_map_COA_tmp[idx_my_region] = np.copy(ssh_COA[i_xs])
    ssh_map_COA_tmp = np.reshape(ssh_map_COA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    ssh_map_COA_tmp = ma.masked_values(ssh_map_COA_tmp,0)
    var_map_COA_tmp = np.zeros(41*41)
    var_map_COA_tmp[idx_my_region] = var_xs_COA[i_xs]
    var_map_COA_tmp = np.reshape(var_map_COA_tmp,(lon_OCCIPUT.shape[0],lon_OCCIPUT.shape[1]))
    var_map_COA_tmp = ma.masked_values(var_map_COA_tmp,0)



    err_map_AnDA_tmp = np.copy(ssh_map_true_tmp)*np.nan
    err_map_AnDA_tmp[15:31,0:26] = ssh_map_AnDA_tmp[15:31,0:26] - ssh_map_true_tmp[15:31,0:26] 

    err_map_OI_tmp = np.copy(ssh_map_true_tmp)*np.nan
    err_map_OI_tmp[15:31,0:26] = ssh_map_OI_tmp[15:31,0:26] - ssh_map_true_tmp[15:31,0:26]

    err_map_COA_tmp = np.copy(ssh_map_true_tmp)*np.nan
    err_map_COA_tmp[15:31,0:26] = ssh_map_COA_tmp[15:31,0:26] - ssh_map_true_tmp[15:31,0:26]

    ke_map_true_tmp = np.copy(ssh_map_true_tmp)*np.nan
    ke_map_true_tmp[15:31,0:26] = np.sqrt(np.copy(ke_true_c[i_xs]))

    ke_map_AnDA_tmp = np.copy(ssh_map_true_tmp)*np.nan
    ke_map_AnDA_tmp[15:31,0:26] = np.sqrt(np.copy(ke_AnDA_c[i_xs]))

    ke_map_OI_tmp = np.copy(ssh_map_true_tmp)*np.nan
    ke_map_OI_tmp[15:31,0:26] = np.sqrt(np.copy(ke_OI_c[i_xs]))

    ke_map_COA_tmp = np.copy(ssh_map_true_tmp)*np.nan
    ke_map_COA_tmp[15:31,0:26] = np.sqrt(np.copy(ke_COA_c[i_xs]))
 
    fig3 = plt.figure(figsize=(20,20))

    fig3.add_subplot(221)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,ssh_map_true_tmp,vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT.flatten()[idx_FLcoast], lat_OCCIPUT.flatten()[idx_FLcoast],np.arange(len(idx_FLcoast))+1000, 20,'o',vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT.flatten()[idx_YCcoast], lat_OCCIPUT.flatten()[idx_YCcoast],np.arange(len(idx_YCcoast))+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([vmin,vmax])
    plt.title('Truth',size=30)
    plt.ylabel('SSH',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[-0.4, 0, 0.4,0.8,1.2])
    cb.ax.set_xticklabels(['-0.4','0', '0.4', '0.8','1.2'],fontsize = 20)  

    fig3.add_subplot(222)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,ssh_map_true_tmp*np.nan,vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_sa[i_sa],lat_sa[i_sa],ssh_sa[i_sa],20,'o',vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([vmin,vmax])
    plt.title('Simulated observations +/- 3 days',size=30)
    cb=m.colorbar(location='bottom',ticks=[-0.4, 0, 0.4,0.8,1.2])
    cb.ax.set_xticklabels(['-0.4','0', '0.4', '0.8','1.2'],fontsize = 20)  


    fig3.add_subplot(223)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,ssh_map_AnDA_tmp,vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([vmin,vmax])
    plt.title('AnDA',size=30)
    cb=m.colorbar(location='bottom',ticks=[-0.4, 0, 0.4,0.8,1.2])
    cb.ax.set_xticklabels(['-0.4','0', '0.4', '0.8','1.2'],fontsize = 20)  

    fig3.add_subplot(224)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,ssh_map_OI_tmp,vmin,vmax,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([vmin,vmax])
    plt.title('OI',size=30)
    cb=m.colorbar(location='bottom',ticks=[-0.4, 0, 0.4,0.8,1.2])
    cb.ax.set_xticklabels(['-0.4','0', '0.4', '0.8','1.2'],fontsize = 20)  

    plt.savefig(path_for_savefigs + 'fig3_' + str(i_xs) + '.png',bbox_inches = 'tight')
    plt.close('all')

    fig4 = plt.figure(figsize=(20,15))
    ax5=fig4.add_subplot(231)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(np.sqrt(var_map_AnDA[i_xs-d_mean:i_xs+d_mean+1]),0),0.,vmax_stdev_AnDA,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_stdev_AnDA])
    cb=m.colorbar(location='bottom',ticks=[0, vmax_stdev_AnDA])
    cb.ax.set_xticklabels(['0', str(vmax_stdev_AnDA)],fontsize = 20)  # vertically oriented colorbar
    plt.ylabel('Estimated standard deviation',fontsize = 30)
    plt.title('AnDA',fontsize = 30)


    fig4.add_subplot(232)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(np.sqrt(var_map_OI[i_xs-d_mean:i_xs+d_mean+1]),0),0.,vmax_stdev_OI,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_stdev_OI])
    cb=m.colorbar(location='bottom',ticks=[0, vmax_stdev_OI])
    cb.ax.set_xticklabels(['0', str(vmax_stdev_OI)],fontsize = 20)  # vertically oriented colorbar
    plt.title('OI',fontsize = 30)

    fig4.add_subplot(233)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(np.sqrt(var_map_COA[i_xs-d_mean:i_xs+d_mean+1]),0),0.,vmax_stdev_COA,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_stdev_COA])
    cb=m.colorbar(location='bottom',ticks=[0, vmax_stdev_COA])
    cb.ax.set_xticklabels(['0', str(vmax_stdev_COA)],fontsize = 20)  # vertically oriented colorbar
    plt.title('OI$_{COA}$',fontsize = 30)

    ax5=fig4.add_subplot(234)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(abserr_map_AnDA[i_xs-d_mean:i_xs+d_mean+1],0),0,vmax_err_AnDA,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_err_AnDA])
    plt.ylabel('Absolute error',fontsize = 30)
    plt.title('AnDA',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[0,vmax_err_AnDA])
    cb.ax.set_xticklabels(['0', str(vmax_err_AnDA)],fontsize = 20)  # vertically oriented colorbar

    fig4.add_subplot(235)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(abserr_map_OI[i_xs-d_mean:i_xs+d_mean+1],0),0.,vmax_err_OI,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_err_OI])
    cb=m.colorbar(location='bottom',ticks=[0, vmax_err_OI])
    cb.ax.set_xticklabels(['0', str(vmax_err_OI)],fontsize = 20)  # vertically oriented colorbar
    plt.title('OI',fontsize = 30)

    fig4.add_subplot(236)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.mean(abserr_map_COA[i_xs-d_mean:i_xs+d_mean+1],0),0.,vmax_err_COA,'YlOrRd')
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,'YlOrRd')
    plt.clim([0.,vmax_err_COA])
    cb=m.colorbar(location='bottom',ticks=[0, vmax_err_COA])
    cb.ax.set_xticklabels(['0', str(vmax_err_COA)],fontsize = 20)  # vertically oriented colorbar
    plt.title('OI$_{COA}$',fontsize = 30)

    fig4.suptitle( month + day + ', ' + str(year),fontsize = 30)

    plt.savefig(path_for_savefigs + 'fig4_' + str(i_xs) + '.png',bbox_inches = 'tight')
    plt.close('all')
     



    fig3 = plt.figure(figsize=(20,20))
    fig3.add_subplot(231)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,vort_map_true[i_xs],-0.08,0.08,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([-0.08,0.08])
    plt.title('Truth',size=30)
    plt.ylabel('vorticity',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[-0.08, -0.04, 0,0.04,0.08])
    cb.ax.set_xticklabels(['-0.08','-0.04', '0','0.04','0.08'],fontsize = 20)  

    fig3.add_subplot(232)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,vort_map_AnDA[i_xs],-0.08,0.08,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([-0.08,0.08])
    plt.title('AnDA',size=30)
    plt.ylabel('vorticity',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[-0.08, -0.04, 0,0.04,0.08])
    cb.ax.set_xticklabels(['-0.08','-0.04', '0','0.04','0.08'],fontsize = 20)  

    fig3.add_subplot(233)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,vort_map_OI[i_xs],-0.08,0.08,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([-0.08,0.08])
    plt.title('OI',size=30)
    plt.ylabel('vorticity',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[-0.08, -0.04, 0,0.04,0.08])
    cb.ax.set_xticklabels(['-0.08','-0.04', '0','0.04','0.08'],fontsize = 20)  


    fig3.add_subplot(235)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.abs(vort_map_AnDA[i_xs] - vort_map_true[i_xs]),0.,0.02,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([0.,0.02])
    plt.title('AnDA',size=30)
    plt.ylabel('vorticity error',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[0,0.01,0.02])
    cb.ax.set_xticklabels(['0','0.01','0.02'],fontsize = 20)  

    fig3.add_subplot(236)
    m=mi.m_pcolor1(lon_OCCIPUT,lat_OCCIPUT,np.abs(vort_map_OI[i_xs] - vort_map_true[i_xs]),0,0.02,mycmap)
    mi.m_scatter1(m,lon_OCCIPUT[20,20:21], lat_OCCIPUT[20,20:21],np.arange(1)+1000, 20,'o',vmin,vmax,mycmap)
    plt.clim([0.,0.02])
    plt.title('OI',size=30)
    plt.ylabel('vorticity error',fontsize = 30)
    cb=m.colorbar(location='bottom',ticks=[0,0.01,0.02])
    cb.ax.set_xticklabels(['0','0.01','0.02'],fontsize = 20)  

    plt.savefig(path_for_savefigs + 'vort_fig4_' + str(i_xs) + '.png',bbox_inches = 'tight')
    plt.close('all')

