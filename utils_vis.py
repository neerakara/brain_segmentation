# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np

# ==========================================================
# ==========================================================       
def save_sample_prediction_results(x,
                                   y_pred,
                                   savepath,
                                   num_rotations=1):
    
    ids = np.arange(24, 128-24, (128-48)//8)
    nc = len(ids)
    nr = 2    
    
    # add one pixel of each label to each slice for consistent visualization
    y_pred_ = np.copy(y_pred)
    for i in range(ids.shape[0]):
        for j in range(15):
            y_pred_[ids[i], 0, j] = j

    plt.figure(figsize=[3*nc, 3*nr])
    for c in range(nc):         
        x_vis = np.rot90(x[ids[c], :, :], k=num_rotations)
        y_pred_vis = np.rot90(y_pred_[ids[c], :, :], k=num_rotations)
        plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_vis, cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('Image')
        plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(y_pred_vis, cmap='tab20'); plt.colorbar(); plt.title('Label')
        
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================       
def save_single_image_and_label(x,
                                y,
                                savepath):
        
    # add one pixel of each label to each slice for consistent visualization
    y_ = np.copy(y)
    for j in range(15):
        y_[0, j] = j

    plt.figure(figsize=[20, 10])      
    plt.subplot(121); plt.imshow(x, cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('Image')
    plt.subplot(122); plt.imshow(y_, cmap='tab20'); plt.colorbar(); plt.title('Label')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()