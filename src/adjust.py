import numpy as np


def adjust_mask(foreground_proposals, predicted_masks, preds_per_image, batchlen, origsize = 512):
    
    foreground_proposals = np.rint(foreground_proposals).astype(np.int32)
    fullimage_mask = np.zeros((batchlen, origsize, origsize, 1))