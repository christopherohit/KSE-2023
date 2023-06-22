import tensorflow as tf
import numpy as np


def roi_align(batch_of_featuremaps, proposals, size):
    """
    The function takes a batch of feature maps, proposals, and a size parameter, and returns a tensor of
    cropped and resized regions of interest (ROIs) from the feature maps based on the proposals.
    
    @param batch_of_featuremaps: A batch of feature maps, typically obtained from a convolutional neural
    network (CNN) applied to an input image
    
    @param proposals: The proposals parameter is a numpy array that contains the proposed regions of
    interest (ROIs) for each image in the batch. Each ROI is represented as a set of four coordinates
    (x1, y1, x2, y2) that define the bounding box of the region. The shape of
    
    @param size: The size parameter is a tuple containing the desired output size of the ROI (region of
    interest) after it has been cropped and resized. It has the format (height, width)
    
    @return: a numpy array of shape (batchlen, proposal_count, size[0], size[1], depth) containing the
    RoI (Region of Interest) feature maps for each proposal in each image in the batch.
    """
    
    batchlen = proposals.shape[0]
    proposal_count = proposals.shape[1]
    depth = batch_of_featuremaps.shape[-1]
    allrois = np.zeros((batchlen, proposal_count, size[0], size[1], depth))
    for image in range(batchlen):
        featuremap = batch_of_featuremaps[image:image + 1]
        proposal = proposals[image]
        proposal = proposal[:]/512
        allrois[image] = tf.image.crop_and_resize(featuremap, proposal, tf.zeros([tf.shape(proposal)[0]], dtype = tf.int32), size)
    return allrois

def mask_roi_aligh(batch_of_featuremaps, batch_of_mask, proposals, size):
    """
    This function takes in a batch of feature maps, masks, proposals, and size, and returns cropped and
    resized feature maps and masks based on the proposals.
    
    @param batch_of_featuremaps: A batch of feature maps extracted from a convolutional neural network
    (CNN) for each image in the batch
    
    @param batch_of_mask: A batch of binary masks for each image in the batch. The shape of the tensor
    should be (batch_size, height, width, 1)
    
    @param proposals: The proposals parameter is a tensor containing the proposed regions of interest
    (ROIs) for each image in the batch. It has a shape of (batch_size, num_proposals, 4), where the last
    dimension represents the coordinates of the top-left and bottom-right corners of each proposal
    
    @param size: The size parameter is a tuple containing the height and width of the output feature
    map. It is used to resize the cropped regions of the input feature map and mask to a fixed size
    
    @return: two numpy arrays: allrois and maskrois.
    """
    
    batchlen = proposals.shape[0]
    proposal_count = proposals.shape[1]
    depth = batch_of_featuremaps.shape[-1]
    
    mask_size = [size[0] * 2, size[1] * 2]
    
    allrois = np.zeros((batchlen, proposal_count, size[0], size[1], depth))
    maskrois = np.zeros((batchlen, proposal_count, mask_size[0], mask_size[1], 1))
    for image in range(batchlen):
        featuremap = batch_of_featuremaps[image: image + 1]
        mask = batch_of_mask[image: image + 1]
        proposal = proposals[image]
        proposal = proposal[:]/512
        allrois[image] = tf.image.crop_and_resize(featuremap, proposal, tf.zeros([tf.shape(proposal)[0]], dtype = tf.int32),size)
        maskrois[image] = tf.image.crop_and_resize(mask, proposal, tf.zeros([tf.shape(proposal)[0]], dtype = tf.int32),size)

    return allrois, maskrois