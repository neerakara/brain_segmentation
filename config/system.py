import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# data dirs
# ==================================================================
orig_data_root_hcp = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
preproc_folder_hcp = '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/data/preprocessed/hcp/'
log_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/simple_segmentation/logdir/'
