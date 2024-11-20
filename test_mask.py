import tensorflow as tf
import os
from model import SPNet, DCCNN, DCCNNSP
from dataGeneration.dataset_tfrecord import get_dataset
import argparse
import scipy.io as scio
import numpy as np
from datetime import datetime
import time
from tools.tools import fft2c_mri, mse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', metavar='str', nargs=1, default=['test'], help='training or test')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['5'], help='number of network iterations')
    parser.add_argument('--subspace_factor', metavar='int', nargs=1, default=['100'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['6'], help='accelerate rate')
    parser.add_argument('--mask_pattern', metavar='str', nargs=1, default=['radial'], help='mask pattern: cartesian, radial, spiral, vista, gauss')
    parser.add_argument('--net', metavar='str', nargs=1, default=['DCCNN'], help='SP_Net, DCCNN')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['7'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['dHCP_2D'], help='dataset name: dHCP_2D, ours_CH_2D, ours_UH_2D, ours_FH_2D, ours_HCP_2D')
    parser.add_argument('--num_trainData', metavar='int', nargs=1, default=['700'], help='number of training data')
    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    dataset_name = args.data[0]
    mode = args.mode[0]
    batch_size = int(args.batch_size[0])
    niter = int(args.niter[0])
    acc = int(args.acc[0])
    mask_pattern = args.mask_pattern[0]
    net_name = args.net[0]
    num_trainData = int(args.num_trainData[0])
    subspace_factor = int(args.subspace_factor[0])
    

    weight_file = 'models/stable/DCCNN_iter_5_acc_6.0_1Dgauss_dHCP_2D_NumData_700/epoch-200/ckpt'
   
    print('network: ', net_name)
    print('acc: ', acc) 
    print('load weight file from: ', weight_file)
    
    result_dir = os.path.join('results', net_name +'_training_on_R6_1Dgauss_NumData_700_test_on_R'+str(acc)+'_'+str(mask_pattern)+'_'+dataset_name)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if dataset_name == 'ours_CH_2D':
        label_dir = os.path.join('results', 'label_CH')
    if dataset_name == 'ours_UH_2D':
        label_dir = os.path.join('results', 'label_UH')
    if dataset_name == 'ours_FH_2D':
        label_dir = os.path.join('results', 'label_FH')
    if dataset_name == 'ours_HCP_2D':
        label_dir = os.path.join('results', 'label_HCP')
    if dataset_name == 'dHCP_2D':
        label_dir = os.path.join('results', 'label')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, TIMESTAMP + net_name + str(acc) + '/'))

    # prepare dataset
    
    mask_size = '274_202'
    dataset = get_dataset(mode, dataset_name, batch_size, shuffle=False)
    tf.print('dataset loaded.')

    # prepare undersampling mask
    mask = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/mask/mask_274_202_'+mask_pattern+'_R'+str(acc)+'.mat')['mask']
    mask = tf.cast(tf.constant(mask), tf.complex64)

   
    
    # initialize network
    if net_name == 'SP_Net':
        Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/UFirst2000.mat')['UFirst2000'][:, 0:200]
        Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/S.mat')['S'][0:200]
        Subspace_total = tf.convert_to_tensor(Subspace_total, dtype=tf.float32) # [55348, 2000]
        Singular_value = tf.convert_to_tensor(np.diag(Singular_value), dtype=tf.float32) # [3500]
        net = SPNet(mask, niter, Subspace_total, Singular_value, subspace_factor)
    elif net_name == 'DCCNN':
        net = DCCNN(mask, niter)
    if net_name == 'DCCNNSP':
        net = DCCNNSP(mask, niter, Subspace_total, Singular_value, subspace_factor)

    net.load_weights(weight_file)
    
    # Iterate over epochs.
    for i, sample in enumerate(dataset):
        # forward
        
        label = sample
                
        label = tf.complex(label, tf.zeros_like(label))
        k0 = fft2c_mri(label)
        k0 = k0 * mask
        label_abs = tf.abs(label)
  
        t0 = time.time()
        recon = net(k0)
        t1 = time.time()
    
        recon_abs = tf.abs(recon)
        label_abs = tf.abs(label)

        loss_total = mse(recon_abs, label_abs)

        tf.print(i, 'mse =', loss_total.numpy(), 'time = ', t1-t0)

        result_file = os.path.join(result_dir, 'recon_'+str(i+1)+'.mat')
        datadict = {'recon': tf.squeeze(recon_abs).numpy()}
        scio.savemat(result_file, datadict)

        
        label_file = os.path.join(label_dir, 'label_'+str(i+1)+'.mat')
        datadict = {'label': tf.squeeze(label_abs).numpy()}
        scio.savemat(label_file, datadict)
        
        
        
        """
        # record gif
        with summary_writer.as_default():
            label_abs = tf.expand_dims(label_abs, axis=-1)
            recon_abs = tf.expand_dims(recon_abs, axis=-1)
            tf.summary.image('image/label', label_abs[0:1, :, :, :].numpy(), step=i)
            tf.summary.image('image/recon', recon_abs[0:1, :, :, :].numpy(), step=i)
        """