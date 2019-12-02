# module for image processing
import numpy as np
import sys
sys.path.append('/home/zhenyicun/AnDA/satellite/codes/')
from mpl_toolkits.basemap import *
import matplotlib.pyplot as mpl
import pandas as pd

# function to compute the velocity field from an image



def compute_U_V_image(image):
    
    # compute spatial derivative    
    U_V=np.gradient(image)
    
    # derive velocity field
    U=-U_V[0]
    V=U_V[1]
    
    # return results
    return U, V

# function to compute the norm of the spatial gradients in an image
def norm_grad_image(image):

    # compute spatial derivative
    grad=np.gradient(image)

    # compute norm of the gradients
    norm_grad=np.hypot(grad[0],grad[1])
    
    # return results
    return norm_grad     
 
# function to plot a gridded satellite image with geographical coordinates
def m_pcolor1(LON,LAT,SST,vmin1,vmax1,colormap):
    
    # Mercator projection
    m = Basemap(projection='merc',llcrnrlat=np.nanmin(LAT),urcrnrlat=np.nanmax(LAT),\
                llcrnrlon=np.nanmin(LON),urcrnrlon=np.nanmax(LON),lat_0=(np.nanmax(LAT)+np.nanmin(LAT))*0.5, \
                lon_0=(np.nanmax(LON)+np.nanmin(LON))*0.5,resolution='c')
    m.drawcoastlines()
    m.fillcontinents()
    m.drawmapboundary(fill_color='aqua')
    m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)
    m.drawparallels(np.arange(10,70,6),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-100,0,6),labels=[0,0,0,1])
    # transform lon & lat 
    x, y = m(LON,LAT)

    # create mask nan values
    sst = ma.masked_where(np.isnan(SST),SST)
    
    # draw map
    m.pcolormesh(x,y,sst,shading='flat',vmin = vmin1,vmax = vmax1,cmap=colormap)
    
    # add coastines

    # return the projection map
    return m
    
    # draw parallels
    #parallels = np.arange(min(LAT.ravel()),max(LAT.ravel()),5.)
    #m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

    # draw meridians
    #meridians = np.arange(min(LON.ravel()),max(LON.ravel()),5.)
    #m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    # fix geographical limits
    #plt.xlim([min(x),max(x)])
    #plt.ylim([min(y),max(y)])
    
# function to plot along-track satellite data with geographical coordinates

def m_pcolor(LON,LAT,SST,colormap):
    
    # Mercator projection
    m = Basemap(projection='merc',llcrnrlat=np.nanmin(LAT),urcrnrlat=np.nanmax(LAT),\
                llcrnrlon=np.nanmin(LON),urcrnrlon=np.nanmax(LON),lat_0=(np.nanmax(LAT)+np.nanmin(LAT))*0.5, \
                lon_0=(np.nanmax(LON)+np.nanmin(LON))*0.5,resolution='c')

    # transform lon & lat 
    x, y = m(LON,LAT)

    # create mask nan values
    sst = ma.masked_where(np.isnan(SST),SST)
    
    # draw map
    m.pcolormesh(x,y,sst,shading='flat',cmap=colormap)
    
    # add coastines
    m.drawcoastlines()
    m.fillcontinents()
    
    # return the projection map
    return m
    
    # draw parallels
    #parallels = np.arange(min(LAT.ravel()),max(LAT.ravel()),5.)
    #m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

    # draw meridians
    #meridians = np.arange(min(LON.ravel()),max(LON.ravel()),5.)
    #m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    # fix geographical limits
    #plt.xlim([min(x),max(x)])
    #plt.ylim([min(y),max(y)])
    
# function to plot along-track satellite data with geographical coordinates
def m_scatter1(m,lon,lat,ssh,size,markertype,vmin1,vmax1,colormap):

    ## transform lon & lat
    x, y = m(lon,lat)

    # apply mean when various measurements at the same position
    data_tmp={'lon' : x, 'lat' : y, 'val' : ssh}
    dataframe_tmp=pd.DataFrame(data_tmp)
    x_merged=dataframe_tmp.groupby(['lon','lat'])['lon'].mean().values
    y_merged=dataframe_tmp.groupby(['lon','lat'])['lat'].mean().values
    ssh_merged=dataframe_tmp.groupby(['lon','lat'])['val'].mean().values
    
    # scatterplot
    m.scatter(x_merged,y_merged,s=size,c=ssh_merged,marker=markertype,vmin = vmin1,vmax = vmax1,cmap=colormap)

def m_scatter(m,lon,lat,ssh,size,markertype,colormap):

    ## transform lon & lat
    x, y = m(lon,lat)

    # apply mean when various measurements at the same position
    data_tmp={'lon' : x, 'lat' : y, 'val' : ssh}
    dataframe_tmp=pd.DataFrame(data_tmp)
    x_merged=dataframe_tmp.groupby(['lon','lat'])['lon'].mean().values
    y_merged=dataframe_tmp.groupby(['lon','lat'])['lat'].mean().values
    ssh_merged=dataframe_tmp.groupby(['lon','lat'])['val'].mean().values
    
    # scatterplot
    m.scatter(x_merged,y_merged,s=size,c=ssh_merged,marker=markertype,cmap=colormap)
    
# function to plot a velocity field with geographical coordinates
def m_quiver(m,lon,lat,u,v,lw,col):

    # transform lon & lat 
    x, y = m(lon,lat)

    # quiver plot
    m.quiver(x,y,u,v,color=col,linewidth=lw)

# function to plot a line with geographical coordinates
def m_line(m,min_lon,min_lat,max_lon,max_lat,col):

    # transform lon & lat 
    x_min, y_min = m(min_lon,min_lat)
    x_max, y_max = m(max_lon,max_lat)

    # plot line
    m.plot([x_min,x_max],[y_min,y_max],col)

# function to plot streamlines with geographical coordinates
def m_streamline(m,LON,LAT,u,v,dens):

    # transform lon & lat 
    x, y = m(LON,LAT)
    
    # draw streamline
    m.streamplot(x, y, u, v, linewidth=np.sqrt(u**2+v**2)*40, color='k',density=dens)
