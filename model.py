from operator import inv
from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
import time
import scipy.io as scio
from tools.tools import tempfft, fft2c_mri, ifft2c_mri, Emat_xy
from tensorflow.python.ops import math_ops

class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_f=32, n_out=2):
        super(CNNLayer, self).__init__()
        self.mylayers = []

        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_out, 3, strides=1, padding='same', use_bias=False))
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        if len(input.shape) == 4:
            input2c = tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
        else:
            input2c = tf.concat([tf.math.real(input), tf.math.imag(input)], axis=-1)
        res = self.seq(input2c)
        res = tf.complex(res[:,:,:,:,0], res[:,:,:,:,1])
        
        return res

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res

class CONV_OP_2D(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP_2D, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv2D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res

###### DC-CNN ######
class DCCNN(tf.keras.Model):
    def __init__(self, mask, niter):
        super(DCCNN, self).__init__(name='DCCNN')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter):
            self.celllist.append(DNCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
    
        x_rec = self.E.mtimes(d, inv=True)
        
        for i in range(self.niter):
            x_rec = self.celllist[i](x_rec, d, d.shape)
    
        return x_rec

class DNCell(layers.Layer):

    def __init__(self, input_shape, E, mask):
        super(DNCell, self).__init__()
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        self.E = E
        self.mask = mask

        self.conv_1 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_2 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_3 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_4 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_5 = CONV_OP_2D(n_f=2, ifactivate=False)

    def call(self, x_rec, d, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
       
        x_rec = self.sparse(x_rec, d) 
  
        return x_rec

    def sparse(self, x_rec, d):

        r_n = tf.stack([tf.math.real(x_rec), tf.math.imag(x_rec)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        x_4 = self.conv_4(x_3)
        x_5 = self.conv_5(x_4)
     
        x_rec = x_5 + r_n
        x_rec = tf.complex(x_rec[:, :, :, 0], x_rec[:, :, :, 1])
        x_rec = self.dc_layer(x_rec, d)

        return x_rec

    def dc_layer(self, x_rec, d):

        k_rec = fft2c_mri(x_rec)
        k_rec = (1 - self.mask) * k_rec + self.mask * d
        x_rec = ifft2c_mri(k_rec)

        return x_rec

###### Subspace_Net ######
class SPNet(tf.keras.Model):
    def __init__(self, mask, niter, subspace, singular_value, subspace_factor):
        super(SPNet, self).__init__(name='SPNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.subspace = subspace
        self.singular_value = singular_value
        self.celllist = []
        self.subspace_shape = subspace.shape 
        self.subspace_factor = subspace_factor  

    def build(self, input_shape):
        self.celllist.append(SubspaceCell(self.subspace_shape, self.singular_value, self.subspace_factor))
        for i in range(self.niter):
            self.celllist.append(SPCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        #subspace = tf.cast(tf.complex(self.subspace, tf.zeros_like(self.subspace)), dtype=tf.complex64) # nx*ny, L
        subspace_learned = self.celllist[0](self.subspace, d.shape) 
        Nxy, L = subspace_learned.shape
        alpha_MFU = self.E.mtimes(tf.transpose(tf.reshape(subspace_learned, [ny, nx, L]),perm=[1,0,2]), inv=False, last=False)
        alpha = tf.linalg.matmul(self.pinv(tf.reshape(alpha_MFU, [nx*ny, L])), tf.transpose(tf.reshape(d, [nb, nx*ny]), perm=[1, 0]))
        
        rho = tf.linalg.matmul(subspace_learned, alpha)
        rho = tf.transpose(tf.reshape(rho, [ny, nx, nb]),perm=[2, 1, 0]) # nx, ny, nb
        #rho = rho / tf.cast(tf.reduce_max(tf.abs(rho)), dtype=tf.complex64)

        #scio.savemat('rec_temp.mat', {'rec':rec.numpy()})
        ## Yunpeng reshape test   

        data = [rho, d]
       
        for i in range(1, self.niter+1):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]

        return x_rec
    
    def pinv(self, matrix):
        matrix = matrix.numpy()
        matrix_pinv = np.linalg.pinv(matrix)
        matrix_pinv = tf.convert_to_tensor(matrix_pinv)
        return matrix_pinv

class SubspaceCell(layers.Layer):

    def __init__(self, input_shape, singular_value, subspace_factor):
        super(SubspaceCell, self).__init__()
        if len(input_shape) == 2:
            self.ns, self.nr = input_shape
        
        self.singular_value = singular_value
        self.subspace_factor = subspace_factor
        
        #self.subspace_variable = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='subspace_variable')
        self.subspace_residual_variable = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='subspace_residual_variable') #0.01
        self.subspace_weight = tf.Variable(tf.constant(1.0, shape=[self.nr - subspace_factor, self.nr - subspace_factor]), trainable=True, name='subspace_weight') #1.0
    
    def call(self, subspace, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        subspace_iter = self.learned_subspace(subspace, self.singular_value) # nx*ny, L
        
        return subspace_iter
    
    def learned_subspace(self, subspace, singular_value):
        Ns, Nr = subspace.shape
        """
        strategy 1:
        thres = tf.sigmoid(self.subspace_variable)*singular_value[0]*0.2
        #L = tf.cast(thres, dtype=tf.int64).numpy()
        L = tf.where(singular_value>thres).shape[0]
        subspace_iter = subspace[:, 0:L]
        
        strategy 2:

        subspace_iter = tf.linalg.matmul(subspace, self.subspace_weight)
        subspace_iter = tf.cast(tf.complex(subspace_iter, tf.zeros_like(subspace_iter)), dtype=tf.complex64)
        """
        subspace_principal = subspace[:, 0:self.subspace_factor]
        subspace_residual = subspace[:, self.subspace_factor:]

        subspace_residual = tf.linalg.matmul(subspace_residual, self.subspace_weight) * self.subspace_residual_variable

        subspace_iter = tf.concat([subspace_principal, subspace_residual], axis=-1)
        #subspace_iter = subspace_principal
        subspace_iter = tf.cast(tf.complex(subspace_iter, tf.zeros_like(subspace_iter)), dtype=tf.complex64)

        return subspace_iter

class SPCell(layers.Layer):

    def __init__(self, input_shape, E, mask):
        super(SPCell, self).__init__()
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        self.E = E
        self.mask = mask

        self.conv_1 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_2 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_3 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_4 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_5 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_6 = CONV_OP_2D(n_f=2, ifactivate=False)

        self.lambda_soft = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda')


    def call(self, data, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        rho,  d = data
        
        rho = self.rho_update(rho,  d)    
        data[0] = rho
    
        return data
     
    def rho_update(self, r_n, d):
        if len(r_n.shape) == 3:
            r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - self.lambda_soft))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        rho = x_6 + r_n
        rho = tf.complex(rho[:, :, :, 0], rho[:, :, :, 1])
        rho = self.dc_layer(rho, d)
        return rho

    def dc_layer(self, x_rec, d):

        k_rec = fft2c_mri(x_rec)
        k_rec = (1 - self.mask) * k_rec + self.mask * d
        x_rec = ifft2c_mri(k_rec)

        return x_rec
    
class ADMMNet(tf.keras.Model):
    def __init__(self, mask, niter):
        super(ADMMNet, self).__init__(name='ADMMNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter):
            self.celllist.append(ADMMCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
    
        x_rec = self.E.mtimes(d, inv=True)
        beta = tf.zeros_like(x_rec)
        z = tf.zeros_like(x_rec)
        data = [x_rec, z, beta, d]
        
        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]
    
        return x_rec

class ADMMCell(layers.Layer):

    def __init__(self, input_shape, E, mask):
        super(ADMMCell, self).__init__()
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        self.E = E
        self.mask = mask

        self.conv_1 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_2 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_3 = CONV_OP_2D(n_f=2, ifactivate=False)
        self.conv_4 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_5 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_6 = CONV_OP_2D(n_f=2, ifactivate=False)
        self.lambda_step = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_1')
        self.lambda_step_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_2')
        self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='eta')

    def call(self, data, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
       
        x_rec, z, beta, d = data

        
        z = self.denoise(x_rec, beta)
        beta = self.beta_mid(beta, x_rec, z)
        x_rec = self.sparse(x_rec, z, d, beta)
        
        data[0] = x_rec
        data[1] = z
        data[2] = beta

        return data

    def sparse(self, x_rec, z, d, beta):
        lambda_step = tf.cast(tf.nn.relu(self.lambda_step), tf.complex64)
        lambda_step_2 = tf.cast(tf.nn.relu(self.lambda_step_2), tf.complex64)

        ATAX_cplx = self.E.mtimes(self.E.mtimes(x_rec, inv=False) - d, inv=True)
        r_n = x_rec - tf.math.scalar_mul(lambda_step, ATAX_cplx) + tf.math.scalar_mul(lambda_step_2, x_rec + beta - z)

        r_n = tf.stack([tf.math.real(x_rec), tf.math.imag(x_rec)], axis=-1)
        v = z - beta
        v = tf.stack([tf.math.real(v), tf.math.imag(v)], axis=-1)
        x_0 = tf.concat([v, r_n], axis=-1) 

        x_1 = self.conv_1(x_0)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
    
        x_rec = x_3 + r_n
        x_rec = tf.complex(x_rec[:, :, :, 0], x_rec[:, :, :, 1])
    
        return x_rec
    
    def denoise(self, x_rec, beta):

        r_n = x_rec + beta
        r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_4(r_n)
        x_2 = self.conv_5(x_1)
        x_3 = self.conv_6(x_2)

        x_3 = tf.complex(x_3[:, :, :, 0], x_3[:, :, :, 1])
        z = x_3 + beta + x_rec

        return z
    
    def beta_mid(self, beta, x_rec, z):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        beta = beta + tf.multiply(eta, x_rec - z)
        return beta
              
class PDNet(tf.keras.Model):
    def __init__(self, mask, niter):
        super(PDNet, self).__init__(name='PDNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter):
            self.celllist.append(PDCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
    
        x_rec = self.E.mtimes(d, inv=True)
        primal = tf.zeros_like(x_rec)
        dual = tf.zeros_like(x_rec)
        data = [primal, dual, d]
        
        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]
    
        return x_rec

class PDCell(layers.Layer):

    def __init__(self, input_shape, E, mask):
        super(PDCell, self).__init__()
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        self.E = E
        self.mask = mask

        self.conv_1 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_2 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_3 = CONV_OP_2D(n_f=2, ifactivate=False)
        self.conv_4 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_5 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_6 = CONV_OP_2D(n_f=2, ifactivate=False)

    def call(self, data, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
       
        primal, dual, d = data

        
        dual = self.dual_update(primal, dual, d)
        primal = self.primal_update(primal, dual)
        
        
        data[0] = primal
        data[1] = dual
      
        return data

    def dual_update(self, primal, dual, d):
        

        evalop_cplx = self.E.mtimes(primal, inv=False) 
        evalop = tf.stack([tf.math.real(evalop_cplx), tf.math.imag(evalop_cplx)], axis=-1)
        dual_float = tf.stack([tf.math.real(dual), tf.math.imag(dual)], axis=-1)
        d_float = tf.stack([tf.math.real(d), tf.math.imag(d)], axis=-1)

        x_0 = tf.concat([dual_float, evalop, d_float], axis=-1) 

        x_1 = self.conv_1(x_0)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
    
        dual_float = dual_float + x_3
        dual = tf.complex(dual_float[:, :, :, 0], dual_float[:, :, :, 1])
    
        return dual
    
    def primal_update(self, primal, dual):

        evalop_cplx = self.E.mtimes(dual, inv=True)
        evalop = tf.stack([tf.math.real(evalop_cplx), tf.math.imag(evalop_cplx)], axis=-1)
        primal_float = tf.stack([tf.math.real(primal), tf.math.imag(primal)], axis=-1)
        x_0 = tf.concat([primal_float, evalop], axis=-1)

        x_1 = self.conv_4(x_0)
        x_2 = self.conv_5(x_1)
        x_3 = self.conv_6(x_2)

        primal_float = primal_float + x_3

        primal = tf.complex(primal_float[:, :, :, 0], primal_float[:, :, :, 1])
       
        return primal

class ISTANet(tf.keras.Model):
    def __init__(self, mask, niter):
        super(ISTANet, self).__init__(name='ISTANet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
    
        self.celllist = []
   

    def build(self, input_shape):
        for i in range(self.niter):
            self.celllist.append(ISTACell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        x_rec = self.E.mtimes(d, inv=True)

        data = [x_rec, d]
       
        for i in range(0, self.niter):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]

        return x_rec
    
class ISTACell(layers.Layer):

    def __init__(self, input_shape, E, mask):
        super(ISTACell, self).__init__()
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        self.E = E
        self.mask = mask

        self.conv_1 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_2 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_3 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_4 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_5 = CONV_OP_2D(n_f=32, ifactivate=True)
        self.conv_6 = CONV_OP_2D(n_f=2, ifactivate=False)

        self.lambda_soft = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda')


    def call(self, data, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        rho,  d = data
        
        rho = self.rho_update(rho,  d)    
        data[0] = rho
    
        return data
     
    def rho_update(self, r_n, d):
        if len(r_n.shape) == 3:
            r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - self.lambda_soft))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        rho = x_6 + r_n
        rho = tf.complex(rho[:, :, :, 0], rho[:, :, :, 1])
        rho = self.dc_layer(rho, d)
        return rho

    def dc_layer(self, x_rec, d):

        k_rec = fft2c_mri(x_rec)
        k_rec = (1 - self.mask) * k_rec + self.mask * d
        x_rec = ifft2c_mri(k_rec)

        return x_rec

###### Subspace_Net ######
class ADMMSPNet(tf.keras.Model):
    def __init__(self, mask, niter, subspace, singular_value, subspace_factor):
        super(ADMMSPNet, self).__init__(name='ADMMSPNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.subspace = subspace
        self.singular_value = singular_value
        self.celllist = []
        self.subspace_shape = subspace.shape 
        self.subspace_factor = subspace_factor  

    def build(self, input_shape):
        self.celllist.append(SubspaceCell(self.subspace_shape, self.singular_value, self.subspace_factor))
        for i in range(self.niter):
            self.celllist.append(ADMMCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        #subspace = tf.cast(tf.complex(self.subspace, tf.zeros_like(self.subspace)), dtype=tf.complex64) # nx*ny, L
        subspace_learned = self.celllist[0](self.subspace, d.shape) 
        Nxy, L = subspace_learned.shape
        alpha_MFU = self.E.mtimes(tf.transpose(tf.reshape(subspace_learned, [ny, nx, L]),perm=[1,0,2]), inv=False, last=False)
        alpha = tf.linalg.matmul(self.pinv(tf.reshape(alpha_MFU, [nx*ny, L])), tf.transpose(tf.reshape(d, [nb, nx*ny]), perm=[1, 0]))
        
        rho = tf.linalg.matmul(subspace_learned, alpha)
        rho = tf.transpose(tf.reshape(rho, [ny, nx, nb]),perm=[2, 1, 0]) # nx, ny, nb
        #rho = rho / tf.cast(tf.reduce_max(tf.abs(rho)), dtype=tf.complex64)

        #scio.savemat('rec_temp.mat', {'rec':rec.numpy()})
        ## Yunpeng reshape test  

        beta = tf.zeros_like(rho)
        z = tf.zeros_like(rho)
        data = [rho, z, beta, d] 
       
        for i in range(1, self.niter+1):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]

        return x_rec
    
    def pinv(self, matrix):
        matrix = matrix.numpy()
        matrix_pinv = np.linalg.pinv(matrix)
        matrix_pinv = tf.convert_to_tensor(matrix_pinv)
        return matrix_pinv

class PDSPNet(tf.keras.Model):
    def __init__(self, mask, niter, subspace, singular_value, subspace_factor):
        super(PDSPNet, self).__init__(name='PDSPNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.subspace = subspace
        self.singular_value = singular_value
        self.celllist = []
        self.subspace_shape = subspace.shape 
        self.subspace_factor = subspace_factor  

    def build(self, input_shape):
        self.celllist.append(SubspaceCell(self.subspace_shape, self.singular_value, self.subspace_factor))
        for i in range(self.niter):
            self.celllist.append(PDCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        #subspace = tf.cast(tf.complex(self.subspace, tf.zeros_like(self.subspace)), dtype=tf.complex64) # nx*ny, L
        subspace_learned = self.celllist[0](self.subspace, d.shape) 
        Nxy, L = subspace_learned.shape
        alpha_MFU = self.E.mtimes(tf.transpose(tf.reshape(subspace_learned, [ny, nx, L]),perm=[1,0,2]), inv=False, last=False)
        alpha = tf.linalg.matmul(self.pinv(tf.reshape(alpha_MFU, [nx*ny, L])), tf.transpose(tf.reshape(d, [nb, nx*ny]), perm=[1, 0]))
        
        rho = tf.linalg.matmul(subspace_learned, alpha)
        rho = tf.transpose(tf.reshape(rho, [ny, nx, nb]),perm=[2, 1, 0]) # nx, ny, nb
        #rho = rho / tf.cast(tf.reduce_max(tf.abs(rho)), dtype=tf.complex64)

        #scio.savemat('rec_temp.mat', {'rec':rec.numpy()})
        ## Yunpeng reshape test 

        #primal = tf.zeros_like(rho)
        dual = tf.zeros_like(rho)
        data = [rho, dual, d]  

       
        for i in range(1, self.niter+1):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]

        return x_rec
    
    def pinv(self, matrix):
        matrix = matrix.numpy()
        matrix_pinv = np.linalg.pinv(matrix)
        matrix_pinv = tf.convert_to_tensor(matrix_pinv)
        return matrix_pinv

###### Subspace_Net with DCCNN ######
class DCCNNSP(tf.keras.Model):
    def __init__(self, mask, niter, subspace, singular_value, subspace_factor):
        super(DCCNNSP, self).__init__(name='DCCNNSPNet')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.subspace = subspace
        self.singular_value = singular_value
        self.celllist = []
        self.subspace_shape = subspace.shape 
        self.subspace_factor = subspace_factor  

    def build(self, input_shape):
        self.celllist.append(SubspaceCell(self.subspace_shape, self.singular_value, self.subspace_factor))
        for i in range(self.niter):
            self.celllist.append(DNCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        #subspace = tf.cast(tf.complex(self.subspace, tf.zeros_like(self.subspace)), dtype=tf.complex64) # nx*ny, L
        subspace_learned = self.celllist[0](self.subspace, d.shape) 
        Nxy, L = subspace_learned.shape
        alpha_MFU = self.E.mtimes(tf.transpose(tf.reshape(subspace_learned, [ny, nx, L]),perm=[1,0,2]), inv=False, last=False)
        alpha = tf.linalg.matmul(self.pinv(tf.reshape(alpha_MFU, [nx*ny, L])), tf.transpose(tf.reshape(d, [nb, nx*ny]), perm=[1, 0]))
        
        rho = tf.linalg.matmul(subspace_learned, alpha)
        rho = tf.transpose(tf.reshape(rho, [ny, nx, nb]),perm=[2, 1, 0]) # nx, ny, nb
        #rho = rho / tf.cast(tf.reduce_max(tf.abs(rho)), dtype=tf.complex64)

        #scio.savemat('rec_temp.mat', {'rec':rec.numpy()})
        ## Yunpeng reshape test   
        
        for i in range(1, self.niter+1):
            rho = self.celllist[i](rho, d, d.shape)
        return rho
    
    def pinv(self, matrix):
        matrix = matrix.numpy()
        matrix_pinv = np.linalg.pinv(matrix)
        matrix_pinv = tf.convert_to_tensor(matrix_pinv)
        return matrix_pinv

class SPNet_fixed(tf.keras.Model):
    def __init__(self, mask, niter, subspace, singular_value, subspace_factor):
        super(SPNet_fixed, self).__init__(name='SPNet_fixed')
        self.niter = niter
        self.E = Emat_xy(mask)
        self.mask = mask
        self.subspace = subspace
        self.singular_value = singular_value
        self.celllist = []
        self.subspace_shape = subspace.shape 
        self.subspace_factor = subspace_factor  

    def build(self, input_shape):
        self.celllist.append(SubspaceCell_fixed(self.subspace_shape, self.singular_value, self.subspace_factor))
        for i in range(self.niter):
            self.celllist.append(SPCell(input_shape, self.E, self.mask))
       
    def call(self, d):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        
        nb, nx, ny = d.shape
         
        #subspace = tf.cast(tf.complex(self.subspace, tf.zeros_like(self.subspace)), dtype=tf.complex64) # nx*ny, L
        subspace_learned = self.celllist[0](self.subspace, d.shape) 
        Nxy, L = subspace_learned.shape
        alpha_MFU = self.E.mtimes(tf.transpose(tf.reshape(subspace_learned, [ny, nx, L]),perm=[1,0,2]), inv=False, last=False)
        alpha = tf.linalg.matmul(self.pinv(tf.reshape(alpha_MFU, [nx*ny, L])), tf.transpose(tf.reshape(d, [nb, nx*ny]), perm=[1, 0]))
        
        rho = tf.linalg.matmul(subspace_learned, alpha)
        rho = tf.transpose(tf.reshape(rho, [ny, nx, nb]),perm=[2, 1, 0]) # nx, ny, nb
        #rho = rho / tf.cast(tf.reduce_max(tf.abs(rho)), dtype=tf.complex64)

        #scio.savemat('rec_temp.mat', {'rec':rec.numpy()})
        ## Yunpeng reshape test   

        data = [rho, d]
       
        for i in range(1, self.niter+1):
            data = self.celllist[i](data, d.shape)
        
        x_rec = data[0]

        return x_rec
    
    def pinv(self, matrix):
        matrix = matrix.numpy()
        matrix_pinv = np.linalg.pinv(matrix)
        matrix_pinv = tf.convert_to_tensor(matrix_pinv)
        return matrix_pinv

class SubspaceCell_fixed(layers.Layer):
    def __init__(self, input_shape, singular_value, subspace_factor):
        super(SubspaceCell_fixed, self).__init__()
        if len(input_shape) == 2:
            self.ns, self.nr = input_shape
        
        self.singular_value = singular_value
        self.subspace_factor = subspace_factor
        
    def call(self, subspace, input_shape):
        if len(input_shape) == 3:
            self.nb, self.nx, self.ny = input_shape
        
        subspace_iter = self.learned_subspace(subspace, self.singular_value) # nx*ny, L
        
        return subspace_iter
    
    def learned_subspace(self, subspace, singular_value):
        Ns, Nr = subspace.shape
        """
        strategy 1:
        thres = tf.sigmoid(self.subspace_variable)*singular_value[0]*0.2
        #L = tf.cast(thres, dtype=tf.int64).numpy()
        L = tf.where(singular_value>thres).shape[0]
        subspace_iter = subspace[:, 0:L]
        
        strategy 2:

        subspace_iter = tf.linalg.matmul(subspace, self.subspace_weight)
        subspace_iter = tf.cast(tf.complex(subspace_iter, tf.zeros_like(subspace_iter)), dtype=tf.complex64)
        """
        subspace_principal = subspace[:, 0:self.subspace_factor]
    
        subspace_iter = subspace_principal
        #subspace_iter = subspace_principal
        subspace_iter = tf.cast(tf.complex(subspace_iter, tf.zeros_like(subspace_iter)), dtype=tf.complex64)

        return subspace_iter