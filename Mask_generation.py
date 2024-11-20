from numpy.lib.type_check import imag
import tensorflow as tf
import os
from model import SPNet, SPNet_fixed, DCCNN, ADMMNet, PDNet, ISTANet, ADMMSPNet, PDSPNet, DCCNNSP
from dataGeneration.dataset_tfrecord import get_dataset
import argparse
import scipy.io as scio
import numpy as np
from datetime import datetime
import time
import tools.mymath as mymath
from tools import compressed_sensing as cs
from matplotlib import pyplot as plt


debug_path = '/home/data/ziwenke/code/dHCP_fast_imaging_MRM/debug/'

shape = [218,182]
mask = cs.cartesian_mask((218,182), 6, sample_n=12, centred=True)

plt.figure()
plt.imshow(mask,cmap='gray')
plt.savefig(debug_path+'mask.png')
plt.close()

#scio.savemat('./mask/mask_218_182_1Dgauss_R6_cen12.mat',{'mask':mask})
mask_old = scio.loadmat('./mask/mask_218_182_1Dgauss_R6.mat')['mask']

plt.figure()
plt.imshow(mask_old,cmap='gray')
plt.savefig(debug_path+'mask_old.png')
plt.close()

