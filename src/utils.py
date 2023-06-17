import tensorflow as tf 
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import os
import nrrd


def generate_anchors(featuremap, orig_shape=512, anchor_sizes = [39,46,52,58,65], anchor_ratios=[1], anchor_stride=1):
    """
    This function generates anchor boxes for object detection based on the input feature map, anchor
    sizes, ratios, and stride.
    
    @param featuremap: The feature map is a 3D numpy array that represents the output of a convolutional
    neural network (CNN) layer. It contains the features extracted from the input image
    
    @param orig_shape: The original shape of the input image, defaults to 512 (optional)
    
    @param anchor_sizes: The sizes of the anchors to be generated. These are the heights and widths of
    the bounding boxes that will be used to detect objects in an image
    
    @param anchor_ratios: anchor_ratios are the ratios of the width to height of the anchor boxes. For
    example, if anchor_ratios=[0.5, 1, 2], it means that for each anchor size, there will be three
    anchor boxes with different width to height ratios of 1:2,
    
    @param anchor_stride: anchor_stride is the distance between the centers of two adjacent anchor boxes
    in the feature map. It is used to control the density of anchor boxes in the feature map. A smaller
    anchor stride will result in more anchor boxes being generated, while a larger anchor stride will
    result in fewer anchor boxes being generated, defaults to 1 (optional)
    
    @return: a numpy array of anchor boxes generated based on the input feature map, anchor sizes,
    anchor ratios, anchor stride, and original image shape.
    """
    
    feature_shapes = featuremap.shape[2]
    feature_strides = orig_shape/featuremap.shape[2]
    anchors = []
    
    # All combinations of indices
    x = np.arange(0, feature_shapes, anchor_sizes) * feature_strides
    y = np.arange(0, feature_shapes, anchor_sizes) * feature_strides
    x,y = np.meshgrid(x,y)
    
    # All combinations of indices, and shapes
    width, x = np.meshgrid(anchor_sizes, x)
    height, y = np.meshgrid(anchor_sizes, y)
    
    # Reshape indices and shapes
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    width = width.flatten().reshape((-1, 1))
    height = height.flatten().reshape((-1, 1))
    
    # Create the centers coordinates and shapes for the anchors
    bbox_centers = np.concatenate((y, x), axis= 1)
    bbox_shapes = np.concatenate((height, width), axis= 1)
    
    # Restructure as [y1, x1, y2, x2]
    bboxes = np.concatenate((bbox_centers - bbox_shapes / 2, bbox_centers + bbox_shapes / 2), axis= 1)
    
    # Anchors are created for each feature map
    anchors.append(bboxes)
    print(f"Num of generated anchors: {len(bboxes)}")
    
    anchors = np.concatenate(anchors, axis= 0)
    anchors = anchors
    return anchors

def read_batch(datafolder, maskfolder, jsonfile, batchlen = 5, start = 0):
    """
    This function reads a batch of medical images and their corresponding masks, labels, and bounding
    boxes from a given folder and returns them as numpy arrays.
    
    @param datafolder: The folder path where the input images are stored
    
    @param maskfolder: The folder path where the mask files are stored
    
    @param jsonfile: The `jsonfile` parameter is a dictionary that contains information about the images
    and their corresponding labels and bounding boxes. The keys of the dictionary are the filenames of
    the images, and the values are dictionaries that contain the label and bounding box information
    
    @param batchlen: The number of images to read in each batch, defaults to 5 (optional)
    
    @param start: The starting index of the batch, i.e., the index of the first image to be read in the
    batch, defaults to 0 (optional)
    
    @return: four arrays: x_batch, m_batch, bb_batch, and y_batch. x_batch and m_batch are 3D arrays of
    shape (batchlen, 512, 512) with an additional dimension of size 1 at the end. bb_batch is a list of
    bounding boxes, and y_batch is a 2D array of shape (batchlen, 2
    """
    x_batch = np.zeros((batchlen, 512, 512))
    y_batch = np.zeros((batchlen, 2))
    m_batch = np.zeros((batchlen, 512, 512))
    bb_batch = []
    
    for num, imnum in enumerate(range(start, start + batchlen)):
        filename = str(imnum).zfill(6) + '.nrrd'
        im,h = nrrd.read(os.path.join(datafolder, filename))
        mask,h = nrrd.read(os.path.join(maskfolder, filename))
        x_batch[num] = im
        m_batch[num] = mask
        y_batch[num] = jsonfile[filename]['label']
        bbox = jsonfile[filename]['bbox']
        bb_batch.append(bbox)
        
    x_batch = np.expand_dims(x_batch, -1)
    m_batch = np.expand_dims(m_batch, -1)
    return x_batch, m_batch, bb_batch, y_batch

def draw_bbox(bboxparam):
    """
    The function takes a bounding box parameter and converts it into 4 lines in matplotlib to visualize
    it.
    
    @param bboxparam: The parameter `bboxparam` is a list containing the coordinates of a bounding box
    in the format [min_x, min_y, max_x, max_y]. The function `draw_bbox` converts these coordinates into
    four lines in matplotlib to visualize the bounding box
    
    @return: a list of four sublists, where each sublist contains two elements representing the x and y
    coordinates of a line segment that forms a bounding box.
    """
    
    # Convert the bounding box to 4 lines in matplotlib to visualize it. boundingbox=[min_x,min_y,max_x,max_y]
    #in matplotlib line=start_x,end_x,start_y,end_y
    #so line by line: lowerline=[x1,x2],[y1,y1] #upperline=[x1,x2],[y2,y2] #leftsideline=[x1,x1],[y1,y2] #rightsideline=[x2,x2],[y1,y2]
    
    y1 = bboxparam[0]
    y2 = bboxparam[2]
    x1 = bboxparam[1]
    x2 = bboxparam[3]
    
    boxlines = [x1,x2],[y1,y1],
    [x1,x2],[y2,y2],
    [x1,x1],[y1,y2],
    [x2,x2],[y1,y2]
    return boxlines

def shift_bbox_pixelwise(anchors, predicted_deltas):
    """
    This function takes in a set of anchor boxes and predicted deltas, and returns a batch of shifted
    bounding boxes.
    
    @param anchors: The anchor boxes are the reference boxes used for object detection. They are
    pre-defined boxes of different sizes and aspect ratios that are placed at various locations in the
    image. The model predicts the offset values for these anchor boxes to adjust their position and size
    to better fit the object in the image
    
    @param predicted_deltas: predicted_deltas is a numpy array containing the predicted deltas for each
    anchor box. The shape of the array should be (N, 4), where N is the number of anchor boxes and 4
    represents the 4 predicted deltas for each box (delta_x, delta_y, delta_w, delta
    
    @return: The function `shift_bbox_pixelwise` returns a numpy array of shape `(n, 4)` where `n` is
    the number of anchors/predicted deltas. The array contains the predicted bounding boxes after
    applying the predicted deltas to the anchor boxes. The four columns of the array represent the x1,
    y1, x2, and y2 coordinates of the predicted bounding boxes.
    """
    
    assert len(anchors.shape) == 2, f"Anchor shape must be 2 dimensions. We got: {str(anchors.shape)}"
    assert len(predicted_deltas.shape) == 2, f"Predicted_deltas shape must be 2 dimensions. We got: {str(predicted_deltas.shape)}"
    
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_centerx = anchors[:, 0] + anchor_widths[:]/2
    anchor_centery = anchors[:, 1] + anchor_heights[:]/2
    
    pred_xc = anchor_centerx[:] + predicted_deltas[:, 0]
    pred_yc = anchor_centery[:] + predicted_deltas[:, 1]
    pred_widths = anchor_widths[:] + predicted_deltas[:, 2]
    pred_heights = anchor_heights[:] + predicted_deltas[:, 3]
    
    predx1 = pred_xc[:] - pred_widths[:]/2
    predy1 = pred_yc[:] - pred_heights[:]/2
    predx2 = pred_xc[:] + pred_widths[:]/2
    predy2 = pred_yc[:] + pred_heights[:]/2
    
    batch_of_boxes = np.stack([predx1, predy1, predx2, predy2], axis=1)
    return batch_of_boxes

def shift_bbox_exponential(anchors,predicted_deltas):
    raise NotImplementedError('Exponential anchorbox shifting is not implemented')

def nms(boxes, scores, proposal_count = 20, nms_threshold = 0.7, padding = True):
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(boxes, scores, proposal_count, iou_threshold = 0.5)
    proposals = tf.gather(boxes, selected_indices)
    proposal_scores = tf.gather(scores, selected_indices)
    
    if padding:
        padding = tf.maximum(proposal_count - tf.shape(selected_indices)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals, proposal_scores, selected_indices

def get_proposals(batch_of_pred_scores,batch_of_pred_deltas,anchors,proposal_count=20,mode='pixelwise'):
    
    batchlen = batch_of_pred_scores.shape[0]
    proposals, origanchors = np.zeros((batchlen, proposal_count, 4))
    
    for image in range(batchlen):
        pred_scores = batch_of_pred_scores[image]
        pred_deltas = batch_of_pred_deltas[image]
        # Find where predicted positive boxes
        
        positive_idxs = np.where(np.argmax(pred_scores, axis= -1) == 1)[0]
        positive_anchors = anchors[positive_idxs]
        selected_boxes = tf.gather(pred_deltas, positive_idxs)
        selected_scores = tf.gather(pred_scores, positive_idxs)
        selected_scores = selected_scores[:, 1]
        
        # Get the predicted anchors for the positive anchors
        if mode == 'pixelwise':
            predicted_boxes = shift_bbox_pixelwise(positive_anchors, selected_boxes)
        else:
            predicted_boxes = shift_bbox_exponential(positive_anchors, selected_boxes)
            
        sorted_indicates = tf.argsort(selected_scores, direction = 'DESCENDING')
        sorted_boxes = tf.cast(tf.gather(predicted_boxes, sorted_indicates), tf.float32)
        sorted_scores = tf.gather(selected_scores, sorted_indicates)
        # sorted_anchors = tf.cast(tf.gather(positive_anchors, sorted_indicates), tf.float32)
        
        proposals[image],_,_ = nms(sorted_boxes, sorted_scores, proposal_count)
        # origanchors[image] = nms(sorted_anchors, sorted_scores, proposal_count)
        
    return proposals
    
def freeze(model):
    for l in model.layers:
        l.trainable = False
        
def unfreeze(model):
    for l in model.layers:
        l.trainable = True
        
