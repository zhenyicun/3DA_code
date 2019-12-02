
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
path_for_save = cwd+'/data/catalog/'


path_OCCIPUT = cwd+'/data/OCCIPUT/'
U = np.load(path_OCCIPUT + 'U.npy')
ssh_mean = np.load(path_OCCIPUT + 'ssh_mean.npy')

max_mode = 100

analogs = np.zeros((1,max_mode))
successors = np.zeros((1,max_mode))
for i in tqdm(range(2,51)):
    ssh_tmp = np.load(path_OCCIPUT + 'ssh_' + str(i) + '.npy')
    ssh_pert_tmp = ssh_tmp - np.repeat(ssh_mean[np.newaxis],ssh_tmp.shape[0],axis = 0)
    ssh_pc_tmp = ssh_pert_tmp.dot(U[:,0:max_mode])
    analogs_tmp = np.copy(ssh_pc_tmp[0:-367])
    successors_tmp = np.copy(ssh_pc_tmp[1:-366])
    analogs = np.vstack((analogs,analogs_tmp))
    successors = np.vstack((successors,successors_tmp))

analogs = np.copy(analogs[1:])
successors = np.copy(successors[1:])



np.save(path_for_save + 'analogs',analogs)
np.save(path_for_save + 'successors',successors)



