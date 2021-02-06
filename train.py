# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
import config.system as sys_config
import scipy.ndimage.interpolation
from skimage import transform

import data.data_hcp as data_hcp

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2l as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.expname_i2l)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.expname_i2l)

    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # if continue_run is set to True, load the model parameters saved earlier
    # else start training from scratch
    # ============================
    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    if exp_config.train_dataset is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 10,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 22,             
                                                              protocol = 'T1',
                                                              size = exp_config.image_size,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    logging.info('Training Images: %s' %str(imtr.shape)) 
    logging.info('Training Labels: %s' %str(gttr.shape))
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')

    utils_vis.save_single_image_and_label(imtr[1, 100, :, :], gttr[1, 100, :, :], log_dir + '/_tr_100.png')
    utils_vis.save_single_image_and_label(imtr[1, :, 100, :], gttr[1, :, 100, :], log_dir + '/_tr_010.png')
    utils_vis.save_single_image_and_label(imtr[1, :, :, 100], gttr[1, :, :, 100], log_dir + '/_tr_001.png')
                
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        image_tensor_shape = [exp_config.batch_size] + [exp_config.image_size[0]] + [exp_config.image_size[1]] + [1]
        label_tensor_shape = [exp_config.batch_size] + [exp_config.image_size[0]] + [exp_config.image_size[1]]
        images_pl = tf.placeholder(tf.float32, shape = image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.uint8, shape = label_tensor_shape, name = 'labels')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, _, _ = model.predict_i2l(images_pl,
                                         exp_config,
                                         training_pl = training_pl)
        
        print('shape of inputs: ', images_pl.shape) # (batch_size, 128, 128, 1)
        print('shape of logits: ', logits.shape) # (batch_size, 128, 128, nlabels)
        
        # ================================================================
        # create a list of all vars that must be optimized wrt
        # ================================================================
        i2l_vars = []
        for v in tf.trainable_variables():
            i2l_vars.append(v)
        
        # ================================================================
        # add ops for calculation of the supervised training loss
        # ================================================================
        loss_op = model.loss(logits,
                             labels_pl,
                             nlabels=exp_config.nlabels,
                             loss_type=exp_config.loss_type_i2l)        
        tf.summary.scalar('loss', loss_op)
        
        # ================================================================
        # add optimization ops.
        # Create different ops according to the variables that must be trained
        # ================================================================
        print('creating training op...')
        train_op = model.training_step(loss_op,
                                       i2l_vars,
                                       exp_config.optimizer_handle,
                                       learning_rate_pl,
                                       update_bn_nontrainable_vars=True)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        print('creating eval op...')
        eval_loss = model.evaluation_i2l(logits,
                                         labels_pl,
                                         images_pl,
                                         nlabels = exp_config.nlabels,
                                         loss_type = exp_config.loss_type_i2l)

        # ================================================================
        # build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        print('creating summary op...')
        summary = tf.summary.merge_all()

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # ================================================================
        # create saver
        # ================================================================
        saver = tf.train.Saver(max_to_keep=10)
        saver_best_dice = tf.train.Saver(max_to_keep=3)
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])
        
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_ops)

        # ================================================================
        # print names of all variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of all variables:' )
        for v in tf.trainable_variables(): print(v.name)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        curr_lr = exp_config.learning_rate
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while (step < exp_config.max_steps):

            if step % 1000 is 0:
                logging.info('============================================================')
                logging.info('step %d' % step)
        
            # ================================================               
            # get random batch
            # ================================================            
            batch = get_random_batch(imtr,
                                     gttr,
                                     batch_size = exp_config.batch_size,
                                     train_or_eval = 'train')
            
            curr_lr = exp_config.learning_rate
            start_time = time.time()
            x, y = batch
                
            # ===========================
            # create the feed dict for this training iteration
            # ===========================
            feed_dict = {images_pl: x,
                         labels_pl: y,
                         learning_rate_pl: curr_lr,
                         training_pl: True}
            
            # ===========================
            # opt step
            # ===========================
            _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

            # ===========================
            # compute the time for this mini-batch computation
            # ===========================
            duration = time.time() - start_time

            # ===========================
            # write the summaries and print an overview fairly often
            # ===========================
            if (step+1) % exp_config.summary_writing_frequency == 0:                    
                logging.info('Step %d: loss = %.3f (%.3f sec for the last step)' % (step+1, loss, duration))
                
                # ===========================
                # Update the events file
                # ===========================
                summary_str = sess.run(summary, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # ===========================
            # Compute the loss on the entire training set
            # ===========================
            if step % exp_config.train_eval_frequency == 0:
                logging.info('Training Data Eval:')
                train_loss, train_dice = do_eval(sess,
                                                 eval_loss,
                                                 images_pl,
                                                 labels_pl,
                                                 training_pl,
                                                 imtr,
                                                 gttr,
                                                 exp_config.batch_size)                    
                
                tr_summary_msg = sess.run(tr_summary,
                                          feed_dict={tr_error: train_loss,
                                                     tr_dice: train_dice})
                
                summary_writer.add_summary(tr_summary_msg, step)
                    
            # ===========================
            # Save a checkpoint periodically
            # ===========================
            if step % exp_config.save_frequency == 0:
                checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

            # ===========================
            # Evaluate the model periodically on a validation set 
            # ===========================
            if step % exp_config.val_eval_frequency == 0:
                logging.info('Validation Data Eval:')
                val_loss, val_dice = do_eval(sess,
                                             eval_loss,
                                             images_pl,
                                             labels_pl,
                                             training_pl,
                                             imvl,
                                             gtvl,
                                             exp_config.batch_size)                    
                
                vl_summary_msg = sess.run(vl_summary,
                                          feed_dict={vl_error: val_loss,
                                                     vl_dice: val_dice})
                
                summary_writer.add_summary(vl_summary_msg, step)

                # ===========================
                # save model if the val dice is the best yet
                # ===========================
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                    saver_best_dice.save(sess, best_file, global_step=step)
                    logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

            # ===========================
            # increment step counter
            # ===========================
            step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    loss_ii = 0
    dice_ii = 0    
    num_eval_batches = 100

    for _ in range(num_eval_batches):
    
        batch = get_random_batch(images,
                                 labels,
                                 batch_size,
                                 train_or_eval = 'eval')

        x, y = batch
        
        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     training_time_placeholder: False}
        
        loss, fg_dice = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_ii += loss
        dice_ii += fg_dice

    avg_loss = loss_ii / num_eval_batches
    avg_dice = dice_ii / num_eval_batches

    logging.info('  Average segmentation loss: %.4f, average dice: %.4f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# function to extract a batch of images and labels
# ==================================================================
def get_random_batch(images,
                     labels,
                     batch_size,
                     train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    subject_indices_this_batch = np.random.randint(0, images.shape[0], size=batch_size)
    
    # train on coronal slices
    if exp_config.train_orientation is 'coronal':
        
        slice_indices_this_batch = np.random.randint(0, images.shape[2], size=batch_size)
        
        x = np.zeros(shape = [batch_size, images.shape[1], images.shape[3]], dtype = images.dtype)
        y = np.zeros(shape = [batch_size, labels.shape[1], labels.shape[3]], dtype = labels.dtype)
        
        for b in range(batch_size):    
            x[b, :, :] = images[subject_indices_this_batch[b], :, slice_indices_this_batch[b], :]
            y[b, :, :] = labels[subject_indices_this_batch[b], :, slice_indices_this_batch[b], :]
    
    # train on axial / transverse slices
    elif exp_config.train_orientation is 'axial':
        
        slice_indices_this_batch = np.random.randint(0, images.shape[3], size=batch_size)
        
        x = np.zeros(shape = [batch_size, images.shape[1], images.shape[2]], dtype = images.dtype)
        y = np.zeros(shape = [batch_size, labels.shape[1], labels.shape[2]], dtype = labels.dtype)
        
        for b in range(batch_size):    
            x[b, :, :] = images[subject_indices_this_batch[b], :, :, slice_indices_this_batch[b]]
            y[b, :, :] = labels[subject_indices_this_batch[b], :, :, slice_indices_this_batch[b]]

    # train on sagittal slices
    elif exp_config.train_orientation is 'sagittal':
        
        slice_indices_this_batch = np.random.randint(0, images.shape[1], size=batch_size)
        
        x = np.zeros(shape = [batch_size, images.shape[2], images.shape[3]], dtype = images.dtype)
        y = np.zeros(shape = [batch_size, labels.shape[2], labels.shape[3]], dtype = labels.dtype)
        
        for b in range(batch_size):    
            x[b, :, :] = images[subject_indices_this_batch[b], slice_indices_this_batch[b], :, :]
            y[b, :, :] = labels[subject_indices_this_batch[b], slice_indices_this_batch[b], :, :]

    # ===========================    
    # data augmentation (contrast changes + random elastic deformations)
    # ===========================      
    if exp_config.da_ratio > 0:

        # ===========================    
        # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
        # ===========================                  
        x, y = do_data_augmentation(images = x,
                                    labels = y,
                                    data_aug_ratio = exp_config.da_ratio,
                                    sigma = exp_config.sigma,
                                    alpha = exp_config.alpha,
                                    trans_min = exp_config.trans_min,
                                    trans_max = exp_config.trans_max,
                                    rot_min = exp_config.rot_min,
                                    rot_max = exp_config.rot_max,
                                    scale_min = exp_config.scale_min,
                                    scale_max = exp_config.scale_max,
                                    gamma_min = exp_config.gamma_min,
                                    gamma_max = exp_config.gamma_max,
                                    brightness_min = exp_config.brightness_min,
                                    brightness_max = exp_config.brightness_max,
                                    noise_min = exp_config.noise_min,
                                    noise_max = exp_config.noise_max)

    x = np.expand_dims(x, axis=-1)
        
    return x, y
        
# ===========================      
# data augmentation: random elastic deformations, translations, rotations, scaling
# data augmentation: gamma contrast, brightness (one number added to the entire slice), additive noise (random gaussian noise image added to the slice)
# ===========================        
def do_data_augmentation(images,
                         labels,
                         data_aug_ratio,
                         sigma,
                         alpha,
                         trans_min,
                         trans_max,
                         rot_min,
                         rot_max,
                         scale_min,
                         scale_max,
                         gamma_min,
                         gamma_max,
                         brightness_min,
                         brightness_max,
                         noise_min,
                         noise_max):
        
    images_ = np.copy(images)
    labels_ = np.copy(labels)
    
    for i in range(images.shape[0]):

        # ========
        # elastic deformation
        # ========
        # if np.random.rand() < data_aug_ratio:
            # images_[i,:,:], labels_[i,:,:] = utils.elastic_transform_image_and_label(images_[i,:,:], labels_[i,:,:], sigma = sigma, alpha = alpha) 

        # ========
        # translation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            
            images_[i,:,:] = scipy.ndimage.interpolation.shift(images_[i,:,:],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 1)
            
            labels_[i,:,:] = scipy.ndimage.interpolation.shift(labels_[i,:,:],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 0)
            
        # ========
        # rotation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_angle = np.random.uniform(rot_min, rot_max)
            
            images_[i,:,:] = scipy.ndimage.interpolation.rotate(images_[i,:,:],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 1)
            
            labels_[i,:,:] = scipy.ndimage.interpolation.rotate(labels_[i,:,:],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 0)
            
        # ========
        # scaling
        # ========
        if np.random.rand() < data_aug_ratio:
            
            n_x, n_y = images_.shape[1], images_.shape[2]
            
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            
            images_i_tmp = transform.rescale(images_[i,:,:], 
                                             scale_val,
                                             order = 1,
                                             preserve_range = True,
                                             mode = 'constant')
            
            
            labels_i_tmp = transform.rescale(labels_[i,:,:],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             mode = 'constant')
            
            images_[i,:,:] = utils.crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
            labels_[i,:,:] = utils.crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)
            
        # ========
        # contrast
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # gamma contrast augmentation
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            images_[i,:,:] = images_[i,:,:]**c
            # not normalizing after the augmentation transformation,
            # as it leads to quite strong reduction of the intensity range when done after high values of gamma augmentation

        # ========
        # brightness
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # brightness augmentation
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            images_[i,:,:] = images_[i,:,:] + c
            
        # ========
        # noise
        # ========
        if np.random.rand() < data_aug_ratio:
            
            # noise augmentation
            n = np.random.normal(noise_min, noise_max, size = images_[i,:,:].shape)
            images_[i,:,:] = images_[i,:,:] + n
            
    return images_, labels_

# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()