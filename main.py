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
from tools.tools import fft2c_mri, mse, extract_batch_from_volume

#tf.debugging.set_log_device_placement(True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['200'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['100'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['5'], help='number of network iterations')
    parser.add_argument('--subspace_factor', metavar='int', nargs=1, default=['100'], help='subspace factor')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['6'], help='accelerate rate')
    parser.add_argument('--mask_pattern', metavar='str', nargs=1, default=['1Dgauss'], help='mask pattern: cartesian, radial, spiral, vista')
    parser.add_argument('--net', metavar='str', nargs=1, default=['PDNet'], help='SP_Net, DCCNN, ADMMNet, PDNet') # Should be adjusted
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['Biobank'], help='dataset name') # Should be adjusted
    parser.add_argument('--num_trainData', metavar='int', nargs=1, default=['1000'], help='number of training data')
    args = parser.parse_args()
    
    # GPU setupimpro
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    #GPUs = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(GPUs[0], True)
    
    mode = 'training'
    dataset_name = args.data[0]
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0]) # PD-Net: 0.0001

    acc = float(args.acc[0])
    mask_pattern = args.mask_pattern[0]
    net_name = args.net[0]
    niter = int(args.niter[0])
    num_trainData = int(args.num_trainData[0])
    subspace_factor = int(args.subspace_factor[0])
    parctice_condition='None'
    
    logdir = '/home/data/ziwenke/code/dHCP_fast_imaging_MRM/logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    model_id  =  net_name + '_iter_'+str(niter) + '_acc_'+ str(acc) + '_' + mask_pattern + '_' + dataset_name + '_NumData_' + str(num_trainData)
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, model_id + '/'))

    modeldir = os.path.join('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/models/stable/', model_id)
    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)

    # prepare undersampling mask
    if dataset_name == 'dHCP':
        mask = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/mask/mask_274_202_'+mask_pattern+'_R'+str(int(acc))+'.mat')['mask']
    elif dataset_name == 'Biobank':
        mask = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/mask/mask_218_182_'+mask_pattern+'_R'+str(int(acc))+'.mat')['mask']
    mask = tf.cast(tf.constant(mask), tf.complex64)

    # prepare dataset
    if dataset_name == 'dHCP_2D':
        multi_coil = False
        mask_size = '274_202'
        dataset = get_dataset(mode, parctice_condition, dataset_name, batch_size, shuffle=False, numData=num_trainData)
        tf.print('dataset loaded.')
    elif dataset_name == 'Biobank':
        multi_coil = False
        mask_size = '218_182'
        dataset = get_dataset(mode, parctice_condition, dataset_name, batch_size, shuffle=False, numData=num_trainData)
        tf.print('dataset loaded.')


    # initialize network
    if net_name == 'SP_Net':
        if dataset_name == 'dHCP_2D':
            Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/UFirst2000.mat')['UFirst2000'][:, 0:200]
            Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/S.mat')['S'][0:200]
        elif dataset_name == 'Biobank':
            Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/Biobank/UFirst700.mat')['UFirst700'][:, 0:200]
            Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/Biobank/S.mat')['S'][0:200]
        Subspace_total = tf.convert_to_tensor(Subspace_total, dtype=tf.float32) # [55348, 2000]
        Singular_value = tf.convert_to_tensor(np.diag(Singular_value), dtype=tf.float32) # [3500]
        net = SPNet(mask, niter, Subspace_total, Singular_value, subspace_factor)
    elif net_name == 'SP_Net_fixed':
        Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/UFirst2000.mat')['UFirst2000'][:, 0:200]
        Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/S.mat')['S'][0:200]
        Subspace_total = tf.convert_to_tensor(Subspace_total, dtype=tf.float32) # [55348, 2000]
        Singular_value = tf.convert_to_tensor(np.diag(Singular_value), dtype=tf.float32) # [3500]
        net = SPNet_fixed(mask, niter, Subspace_total, Singular_value, subspace_factor)
    elif net_name == 'PDSPNet':
        Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/UFirst2000.mat')['UFirst2000'][:, 0:200]
        Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/S.mat')['S'][0:200]
        Subspace_total = tf.convert_to_tensor(Subspace_total, dtype=tf.float32) # [55348, 2000]
        Singular_value = tf.convert_to_tensor(np.diag(Singular_value), dtype=tf.float32) # [3500]
        net = PDSPNet(mask, niter, Subspace_total, Singular_value, subspace_factor)
    elif net_name == 'DCCNNSP':
        Subspace_total = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/UFirst2000.mat')['UFirst2000'][:, 0:200]
        Singular_value = scio.loadmat('/home/data/ziwenke/code/dHCP_fast_imaging_MRM/1-subspace/subspaceEst/eigen_image/slice274_202/S.mat')['S'][0:200]
        Subspace_total = tf.convert_to_tensor(Subspace_total, dtype=tf.float32) # [55348, 2000]
        Singular_value = tf.convert_to_tensor(np.diag(Singular_value), dtype=tf.float32) # [3500]
        net = DCCNNSP(mask, niter, Subspace_total, Singular_value, subspace_factor)
    elif net_name == 'DCCNN':
        net = DCCNN(mask, niter)
    elif net_name == 'ADMMNet':
        net = ADMMNet(mask, niter)
    elif net_name == 'PDNet':
        net = PDNet(mask, niter)
    elif net_name == 'ISTANet':
        net = ISTANet(mask, niter)
    

    tf.print('network initialized.')

    learning_rate_org = learning_rate
    learning_rate_decay = 0.95

    optimizer = tf.optimizers.Adam(learning_rate_org)
    
    # Iterate over epochs.
    total_step = 0
    param_num = 0
    loss = 0

    for epoch in range(num_epoch):
        for step, sample in enumerate(dataset):
            
            # forward
            t0 = time.time()
            with tf.GradientTape() as tape:
                label = sample
                #scio.savemat('T2w.mat', {'T2w':T2w.numpy()})
                
                label = tf.complex(label, tf.zeros_like(label))
                k0 = fft2c_mri(label)
                k0 = k0 * mask
                recon = net(k0)
                recon_abs = tf.abs(recon)
                label_abs = tf.abs(label)

                loss_mse = mse(recon_abs, label_abs)

            # backward
            grads = tape.gradient(loss_mse, net.trainable_weights)####################################
            optimizer.apply_gradients(zip(grads, net.trainable_weights))#################################

            # record loss
            with summary_writer.as_default():
                tf.summary.scalar('loss/total', loss_mse.numpy(), step=total_step)

            # record gif
            
            
            if step % 4 == 0:
                with summary_writer.as_default():
                    label_abs = tf.expand_dims(label_abs, axis=-1)
                    recon_abs = tf.expand_dims(recon_abs, axis=-1)
                    tf.summary.image('image/label', label_abs[0:1, :, :, :].numpy(), step=total_step)
                    tf.summary.image('image/recon', recon_abs[0:1, :, :, :].numpy(), step=total_step)
            
            # calculate parameter number
            if total_step == 0:
                param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])

            # log output
            tf.print('Epoch', epoch+1, '/', num_epoch, 'Step', step, 'loss =', loss_mse.numpy(), 'time', time.time() - t0, 'lr = ', learning_rate, 'param_num', param_num)
            total_step += 1

        # learning rate decay for each epoch
        learning_rate = learning_rate_org * learning_rate_decay ** (epoch + 1)#(total_step / decay_steps)
        optimizer = tf.optimizers.Adam(learning_rate)

        # save model each epoch
        #if epoch in [0, num_epoch-1, num_epoch]:
        model_epoch_dir = os.path.join(modeldir,'epoch-'+str(epoch+1), 'ckpt')
        net.save_weights(model_epoch_dir, save_format='tf')

