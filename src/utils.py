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
    
    :param featuremap: The feature map is a 3D numpy array that represents the output of a convolutional
    neural network (CNN) layer. It contains the features extracted from the input image
    :param orig_shape: The original shape of the input image, defaults to 512 (optional)
    :param anchor_sizes: The sizes of the anchors to be generated. These are the heights and widths of
    the bounding boxes that will be used to detect objects in an image
    :param anchor_ratios: anchor_ratios are the ratios of the width to height of the anchor boxes. For
    example, if anchor_ratios=[0.5, 1, 2], it means that for each anchor size, there will be three
    anchor boxes with different width to height ratios of 1:2,
    :param anchor_stride: anchor_stride is the distance between the centers of two adjacent anchor boxes
    in the feature map. It is used to control the density of anchor boxes in the feature map. A smaller
    anchor stride will result in more anchor boxes being generated, while a larger anchor stride will
    result in fewer anchor boxes being generated, defaults to 1 (optional)
    :return: a numpy array of anchor boxes generated based on the input feature map, anchor sizes,
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


def calculate_ious(bbox, anchors):
    """
    This function calculates the intersection over union (IOU) between a bounding box and a set of
    anchor boxes.
    
    :param bbox: The bbox parameter is a list or array containing the coordinates of a bounding box in
    the format [y1, x1, y2, x2], where y1 and x1 are the coordinates of the top-left corner of the box,
    and y2 and x2 are the coordinates of the
    :param anchors: The anchors parameter is a numpy array containing the coordinates of the anchor
    boxes. Each row of the array represents an anchor box and contains four values: the x-coordinate of
    the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right
    corner, and the y-coordinate
    :return: an array of Intersection over Union (IoU) values between a bounding box and a set of anchor
    boxes.
    """
    
    # area = width * height
    anchorarea = (anchors[:,2] - anchors[:,0]) * (anchors[:,3] - anchors[:,1])
    bboxarea = (bbox[2] - bbox[0]) * (bbox[3] * bbox[1])
    
    y1 = np.maximum(bbox[0], anchors[:, 0])
    y2 = np.minimum(bbox[2], anchors[:, 2])
    x1 = np.maximum(bbox[1], anchors[:, 1])
    x2 = np.minimum(bbox[3], anchors[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = bboxarea + anchorarea[:] - intersection[:]
    iou = intersection / union
    
    return iou

def calculate_pixelwise_deltas(bbox, anchors, numof):
    """
    This function calculates the predicted dx, dy, dw, and dh for each anchor based on the given
    bounding box and anchor parameters.
    
    :param bbox: The bounding box coordinates of an object in the format [xmin, ymin, xmax, ymax]
    :param anchors: A numpy array of shape (numof, 4) containing the coordinates of the anchor boxes.
    Each row represents an anchor box and the four columns represent the x-coordinate of the top-left
    corner, y-coordinate of the top-left corner, x-coordinate of the bottom-right corner, and
    y-coordinate of
    :param numof: The number of anchors
    :return: a numpy array of shape (numof, 4) containing the predicted dx, dy, dw, dh values for each
    anchor.
    """
    assert len(anchors.shape) == 2, f"2 dimenzios anchors shape kell. Kapott: {str(anchors.shape)}"
    assert len(bbox.shape) == 1, f"1 dimenzios bbox shape kell. Kapott: {str(bbox.shape)}"
    
    # Predicted dx,dy,dw,dh for each anchor
    deltas = np.zeros((numof, 4))
    
    anchor_width = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]
    anchor_centerx = anchors[:, 0] + anchor_width[:]/2
    anchor_centery = anchors[:, 1] + anchor_height[:]/2
    
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_centerx = bbox[0] + bbox_width/2
    bbox_centery = bbox[1] + bbox_height/2
    
    dw = bbox_width - anchor_width[:]
    dh = bbox_height - anchor_height[:]
    dx = bbox_centerx - anchor_centerx[:]
    dy = bbox_centery - anchor_centery[:]
    
    for anchor in range(numof):
        deltas[anchor] = [dx[anchor], dy[anchor], dw[anchor], dh[anchor]]
    
    return deltas

def calculate_exponential_deltas(bbox, anchors, numof):
    raise NotImplementedError('Exponential anchorbox shifting is not implemented')

def indices_deltas_labels(batch_of_bboxes, anchors, batchlen, train_set_size = 20, mode = 'pixelwise'):
    
    num_of_anchors = len(anchors)
    batch_of_bboxes = np.array(batch_of_bboxes)
    
    batch_of_indices = np.zeros((batchlen, train_set_size, 2), dtype= np.int32)
    batch_of_deltas = np.zeros((batchlen, num_of_anchors, 4))
    batch_of_labels = np.zeros((batchlen, train_set_size))
    
    for im in range(batchlen):
        bboxes_im = np.asarray(batch_of_bboxes[im])
        num_of_bboxes = bboxes_im.shape[0]
        
        indices = np.zeros((train_set_size, 2), dtype= np.int32)
        deltas = np.zeros((num_of_anchors, 4))
        boxlabels = np.zeros((train_set_size))
        
        if num_of_bboxes > 1:
            
            # Intersection over union score for each bbox-anchor pair
            bbox_ious = np.zeros((num_of_bboxes, num_of_anchors))
            # Desired delta x,y,h,w for each bbox-anchor pair --> RPN shoult predict these
            bbox_deltas = np.zeros((num_of_bboxes, num_of_anchors, 4))
            ious = np.zeros((num_of_anchors))
            
            for bboxnum, bbox in enumerate(bboxes_im):
                if mode == "pixelwise":
                    bbox_deltas[bboxnum] = calculate_pixelwise_deltas(bbox, anchors, num_of_anchors)
                else:
                    bbox_deltas[bboxnum] = calculate_exponential_deltas(bbox, anchors, num_of_anchors)
                
                bbox_ious[bboxnum] = calculate_ious(bbox, anchors)
            
            # We want to train the anchors to move to the nearest bbox, if there are more --> so even if there are more bboxes, we only have one delta/iou value for each anchor
            for anchor in range(num_of_anchors):
                nearest_bbox = np.argmax(bbox_ious[:, anchor])
                deltas[anchor] = bbox_deltas[nearest_bbox, anchor]
                ious[anchor] = bbox_ious[nearest_bbox, anchor]
        else:
            # If there are no masks on the image, the bbox of it is [0,0,0,0]
            if np.all(np.equal(bboxes_im, 0)):
                sampledanchors = random.sample(range(0, num_of_anchors), train_set_size)
                indices = [[im, x] for x in sampledanchors]
                batch_of_indices[im] = indices
                batch_of_deltas[im] = deltas
                batch_of_labels[im] = boxlabels
                continue
            
            else:
                bbox = bboxes_im[0]
                if mode == "pixelwise":
                    deltas = calculate_pixelwise_deltas(bbox, anchors, num_of_anchors)
                else:
                    deltas = calculate_exponential_deltas(bbox, anchors, num_of_anchors)
                ious = calculate_ious(bbox, anchors)
                
        # We choose anchors with IoU>0.5 values to be foreground boxes, with IoU<0.1 to be backround boxes
        num = 0
        bg_indices = []
        for anchor in range(num_of_anchors):
            if ious[anchor] > 0.5:
                indices[num] = [im, anchor]
                if num < train_set_size // 2:
                    num += 1
            elif ious[anchor] < 0.1:
                bg_indices.append(anchor)
        
        sampledanchors = random.sample(bg_indices, train_set_size - num)
        indices[num:] = [[im, x] for x in sampledanchors]
        boxlabels[0:num] = 1
        boxlabels[num:] = 0
        
        batch_of_indices[im] = indices
        batch_of_deltas[im] = deltas
        batch_of_labels[im] = boxlabels
        
    return batch_of_indices, batch_of_deltas, batch_of_labels