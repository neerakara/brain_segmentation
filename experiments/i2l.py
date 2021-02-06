import model_zoo
import tensorflow as tf

# training dataset
train_dataset = 'HCPT1'
tr_str = 'tr' + train_dataset

# slicing orientation while training
train_orientation = 'axial' # 'coronal' or 'axial' or 'transverse'

# Model settings : i2l
model_handle_i2l = model_zoo.unet2D_i2l

# run number
run_number = 1
run_str = '_run' + str(run_number)

# data aug settings
da_ratio = 0.25
sigma = 20
alpha = 1000
trans_min = -10
trans_max = 10
rot_min = -10
rot_max = 10
scale_min = 0.9
scale_max = 1.1
gamma_min = 0.5
gamma_max = 2.0
brightness_min = 0.0
brightness_max = 0.1
noise_min = 0.0
noise_max = 0.1
da_str = '_da' + str(da_ratio)

# ======================================================================
# data settings
# ======================================================================
image_size = (256, 300, 256)
target_resolution_brain = (0.7, 0.7, 0.7)
nlabels = 15
size_str = 'size_' + '_'.join([str(i) for i in image_size])
res_str = 'res_' + '_'.join([str(i) for i in target_resolution_brain])
data_str = size_str + res_str

# exp name
expname_i2l = tr_str + data_str + '_tr_orientation_' + train_orientation + da_str + run_str

# ======================================================================
# training loss
# ======================================================================
loss_type_i2l = 'dice'

# ======================================================================
# training settings
# ======================================================================
max_steps = 50001
batch_size = 16
learning_rate = 1e-3    
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 100
train_eval_frequency = 1000
val_eval_frequency = 1000
save_frequency = 1000
debug = False
continue_run = False