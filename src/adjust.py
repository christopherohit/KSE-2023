import numpy as np
import tensorflow as tf
from utils import nms


def adjust_mask(foreground_proposals, predicted_masks, preds_per_image, batchlen, origsize = 512):
    """
    This function adjusts the predicted masks based on the foreground proposals and returns the full
    image masks.
    
    @param foreground_proposals: A numpy array containing the bounding box coordinates of the foreground
    proposals for each image in the batch. The shape of the array is (num_proposals, 4), where
    num_proposals is the total number of foreground proposals across all images in the batch, and the 4
    columns represent the x-coordinate
    
    @param predicted_masks: an array of predicted masks for each foreground proposal
    
    @param preds_per_image: `preds_per_image` is a list that contains the number of predicted masks for
    each image in the batch. For example, if the batch contains 3 images and the number of predicted
    masks for each image is [2, 1, 3], then there are a total of 6
    
    @param batchlen: The number of images in the batch
   
    @param origsize: The original size of the input image, defaults to 512 (optional)
    
    @return: a numpy array of shape (batchlen, origsize, origsize, 1) containing the adjusted masks for
    the foreground proposals.
    """
    
    foreground_proposals = np.rint(foreground_proposals).astype(np.int32)
    fullimage_masks = np.zeros((batchlen, origsize, origsize, 1), dtype = np.float32)
    num = 0
    for im in range(batchlen):
        for proposal in range(preds_per_image[im]):
            if preds_per_image[im] == 0:
                continue
            
            actualproposal = proposal + num
            x1 = foreground_proposals[actualproposal, 0]
            x2 = foreground_proposals[actualproposal, 2]
            y1 = foreground_proposals[actualproposal, 1]
            y2 = foreground_proposals[actualproposal, 3]
            
            
            if x1 < 0:
                x1 = 0
            if x1 > 0:
                y1 = 0
            if x2 > 511:
                x2 = 511
            if y2 > 511:
                y2 = 511
                
            size = [x2 - x1, y2 - y1]
            
            if np.all(np.equal(size, 0)):
                continue
            
            mask = np.copy(predicted_masks[actualproposal])
            mask = tf.image.resize_with_pad(mask, size[0], size[1])
            fullimage_masks[im][x1:x2, y1:y2, :] = mask
            
        num += preds_per_image[im]
    return fullimage_masks

def adjust_boxes(predicted_refined_boxes, preds_per_image, batchlen):
    """
    The function adjusts predicted refined boxes by rounding them and grouping them by image.
    
    @param predicted_refined_boxes: This is a numpy array containing the predicted bounding boxes for
    all images in a batch. The shape of this array is (total number of predicted boxes, 4), where the
    second dimension represents the coordinates of the top-left and bottom-right corners of the bounding
    box
    
    @param preds_per_image: A list containing the number of predicted boxes for each image in the batch
    
    @param batchlen: The number of images in the batch
    
    @return: a list of adjusted bounding boxes for each image in the batch. The adjusted boxes are
    obtained by rounding the predicted refined boxes and grouping them by image.
    """
    num = 0
    adjust_boxes = []
    for im in range(batchlen):
        first = num
        last = first + preds_per_image[im]
        
        rounded = np.rint(predicted_refined_boxes[first:last])
        adjust_boxes.append(rounded)
        num += preds_per_image[im]
        
    return adjust_boxes

def adjust_scores(predicted_softmax_scores, preds_per_images, batchlen):
    """
    The function adjusts predicted softmax scores and returns the maximum scores and corresponding
    labels for each image in a batch.
    
    @param predicted_softmax_scores: It is a numpy array containing the predicted softmax scores for
    each class for all the images in a batch. The shape of the array is (total number of images in the
    batch * number of classes)
   
    @param preds_per_images: The number of predictions made for each image in the batch
   
    @param batchlen: The number of images in the batch
   
    @return: two lists: `adjusted_scores` and `adjusted_labels`.
    """
    num = 0
    adjusted_scores = []
    adjusted_labels = []
    for im in range(batchlen):
        first = num
        last = first + preds_per_images[im]
        adjusted_scores.append(np.amax(predicted_softmax_scores[first:last], axis= -1))
        adjusted_labels.append(np.argmax(predicted_softmax_scores[first:last], axis= -1))
        
        
    return adjusted_scores, adjusted_labels

def adjust_to_batch(foreground_proposals, predicted_classlabels, predicted_softmax_scores, predicted_refined_boxes,
                    preds_per_images, batchlen, predicted_masks = None):
    
    """
    This function adjusts predicted bounding boxes, scores, and labels to match the batch size and
    performs non-maximum suppression, and optionally adjusts predicted masks.
    
    @param foreground_proposals: It is a tensor containing the foreground proposals generated by the
    Region Proposal Network (RPN) for each image in the batch
    
    @param predicted_classlabels: The predicted class labels for each object proposal in the batch
    
    @param predicted_softmax_scores: The predicted class probabilities for each proposal, outputted by a
    neural network
   
    @param predicted_refined_boxes: The predicted bounding boxes for the objects in the input images,
    after refinement by the object detection model
   
    @param preds_per_images: The number of predicted objects per image in the batch
   
    @param batchlen: The number of images in the batch
   
    @param predicted_masks: A tensor containing the predicted masks for each object proposal in the
    batch
   
    @return: either the adjusted boxes, scores, and labels or the adjusted masks, boxes, scores, and
    labels depending on whether the predicted masks are provided or not.
    """
    
    adjusted_boxes = adjust_boxes(predicted_refined_boxes, preds_per_images, batchlen)
    adjusted_scores, adjusted_labels = adjust_scores(predicted_softmax_scores, preds_per_images, batchlen)
    
    for i in range(len(adjusted_scores)):
        box_proposals, score_proposals, indices = nms(adjusted_boxes[i], adjusted_scores[i], nms_threshold= 0.8, padding= False)
        label_proposals = tf.gather(adjusted_labels[i], indices)
        adjusted_boxes[i] = box_proposals.numpy()
        adjusted_scores[i] = np.around(score_proposals.numpy(), decimals= 3)
        adjusted_labels[i] = label_proposals.numpy()
        
        
    if predicted_masks is None:
        return adjusted_boxes, adjusted_scores, adjusted_labels
    else:
        adjusted_masks = adjust_mask(foreground_proposals, predicted_masks, preds_per_images, batchlen)
        return adjusted_masks, adjusted_boxes, adjusted_scores, adjusted_labels