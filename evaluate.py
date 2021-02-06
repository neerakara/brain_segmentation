# ==================================================================
# @Kerem: Replace loading of the test image (lines 99 to 107) with your images.
# ==================================================================

# ==================================================================
# import 
# ==================================================================
import logging
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
from skimage.transform import rescale
import config.system as sys_config
import data.data_hcp as data_hcp

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2l as exp_config
    
# ==================================================================
# main function for training
# ==================================================================
def predict_segmentation(image):
    
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32,
                                   shape = [None] + [exp_config.image_size[0]] + [exp_config.image_size[1]] + [1],
                                   name = 'images')
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, softmax, preds = model.predict_i2l(images_pl,
                                                   exp_config,
                                                   training_pl = tf.constant(False, dtype=tf.bool))
                                                        
        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver() 
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)
        
        # ================================================================
        # Restore the segmentation network parameters
        # ================================================================
        logging.info('============================================================')        
        path_to_model = sys_config.log_root + exp_config.expname_i2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
                
        # ================================================================
        # predict segmentation
        # ================================================================
        X = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
        mask_predicted = sess.run(preds, feed_dict={images_pl: X})
        mask_predicted = np.squeeze(np.array(mask_predicted)).astype(float)  
        
        sess.close()
        
        return mask_predicted
        
# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test image
    # ===================================      
    data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                           preprocessing_folder = sys_config.preproc_folder_hcp,
                                                           idx_start = 50,
                                                           idx_end = 52,                
                                                           protocol = 'T1',
                                                           size = exp_config.image_size,
                                                           target_resolution = exp_config.target_resolution_brain)
    imts = data_brain_test['images']                
    image = imts[1,:,:,100]  
                            
    # ===================================
    # predict segmentation at the pre-processed resolution
    # ===================================
    predicted_label = predict_segmentation(image)

    # ===================================
    # save sample results
    # ===================================
    utils_vis.save_single_image_and_label(image,
                                            predicted_label,
                                            savepath = sys_config.log_root + exp_config.expname_i2l + '/test_result.png')
        
# ===================================
# ===================================
if __name__ == '__main__':
    main()