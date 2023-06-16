import numpy as np
import random
from calculate import calculate_exponential_deltas, calculate_ious, calculate_pixelwise_deltas



def head_indices_deltas_labels(batch_of_bboxes, batch_of_gt_labels, proposals, batchlen, train_set_size = 6, mode = 'pixelwise'): 
    """
    This function generates indices, deltas, and labels for training a region proposal network (RPN)
    based on the input batch of bounding boxes and ground truth labels.
    
    @param batch_of_bboxes: A list of bounding boxes for each image in the batch. Each bounding box is
    represented as a list of four values: [x, y, width, height]
    
    @param batch_of_gt_labels: batch_of_gt_labels is a numpy array containing the ground truth labels
    for each image in the batch. The shape of the array is (batchlen, num_classes), where batchlen is
    the number of images in the batch and num_classes is the number of classes in the dataset (including
    the background class
    
    @param proposals: proposals is a numpy array containing the proposed bounding boxes for each image
    in the batch. It has shape (batchlen, num_of_proposals, 4), where batchlen is the number of images
    in the batch, num_of_proposals is the number of proposed bounding boxes for each image
    
    @param batchlen: The number of images in the batch
    
    @param train_set_size: The number of samples to be included in each training batch, defaults to 6
    (optional)
    
    @param mode: The mode parameter specifies whether to use "pixelwise" or "exponential" method for
    calculating deltas, defaults to pixelwise (optional)
    
    @return: three numpy arrays: batch_of_indices, batch_of_deltas, and batch_of_labels.
    """
    num_of_proposals = proposals.shape[1]
    batch_of_bboxes = np.array(batch_of_bboxes)
    
    batch_of_indices = np.zeros((batchlen, train_set_size, 2), dtype= np.int32)
    batch_of_deltas = np.zeros((batchlen, num_of_proposals, 4))
    batch_of_labels = np.zeros((batchlen, train_set_size))
    
    for im in range(batchlen):
        bboxes_im = np.asarray(batch_of_bboxes[im])
        num_of_bboxes = bboxes_im.shape[0]
        
        # [0,1], [1,0] when having one bounding box, [1,1] when having two, [0,0] when having 0.
        gt_label = batch_of_gt_labels[im]
        proposal_of_image = proposals[im]
        
        indices = np.zeros((train_set_size, 2), dtype= np.int32)
        deltas = np.zeros((num_of_proposals, 4))
        boxlabels = np.zeros((train_set_size))
        nearest_bboxes = np.zeros((num_of_proposals))
        
        if num_of_bboxes > 1:
            
            # Intersection over union score for each bbox-anchor pair
            bbox_ious = np.zeros((num_of_bboxes, num_of_proposals))
            
            # Desired delta x,y,h,w for each bbox-anchor pair --> RPN shoult predict these
            bbox_deltas = np.zeros((num_of_bboxes, num_of_proposals, 4))
            
            ious = np.zeros((num_of_proposals))
            
            for bboxnum, bbox in enumerate(bboxes_im):
                if mode == "pixelwise":
                    bbox_deltas[bboxnum] = calculate_pixelwise_deltas(bbox, proposal_of_image, num_of_proposals)
                else:
                    bbox_deltas[bboxnum] = calculate_exponential_deltas(bbox, proposal_of_image, num_of_proposals)
                bbox_ious[bboxnum] = calculate_ious(bbox, proposal_of_image)
                
            # We want to train the anchors to move to the nearest bbox, if there are more --> so even if there are more bboxes, we only have one delta/iou value for each anchor
            for proposal in range(num_of_proposals):
                nearest_bbox = np.argmax(bbox_ious[:, proposal])
                nearest_bboxes[proposal] = nearest_bbox
                deltas[proposal] = bbox_deltas[nearest_bbox, proposal]
                ious[proposal] = bbox_ious[nearest_bbox, proposal]
        
        else:
            
            # If there are no masks on the image, the bbox of it is [0,0,0,0]
            if np.all(np.equal(bboxes_im, 0)):
                sampledanchors = random.sample(range(0, num_of_proposals), train_set_size)
                indices = [[im, x] for x in sampledanchors]
                # 2 (numofclasses+1) is the label of background
                boxlabels = boxlabels + 2
                batch_of_indices[im] = indices
                batch_of_deltas[im] = deltas
                batch_of_labels[im] = boxlabels
                continue
            
            else:
                bbox = bboxes_im[0]
                if mode == 'pixelwise':
                    deltas = calculate_pixelwise_deltas(bbox, proposal_of_image, num_of_proposals)
                else:
                    deltas = calculate_exponential_deltas(bbox, proposal_of_image, num_of_proposals)
                
                ious = calculate_ious(bbox, proposal_of_image)
                nearest_bboxes = nearest_bboxes + np.argmax(gt_label)
        
        
        # We choose anchors with IoU>0.5 values to be foreground boxes, with 0.1<IoU<0.5 to be backround boxes
        num = 0
        bg_indices = []
        for proposal in range(num_of_proposals):
            if ious[proposal] > 0.4:
                indices[num] = [im, proposal]
                boxlabels[num] = nearest_bboxes[proposal]
                if num < train_set_size // 2:
                    num += 1
            else:
                bg_indices.append(proposal)
                
        # Around half of the set consists of foreground boxes, half of it will be a randomly sampled set of background boxes
        sampledanchors = random.sample(bg_indices, train_set_size - num)
        indices[num:] = [[im, x] for x in sampledanchors]
        
        # 2 (numofclasses+1) is the label of background
        boxlabels[num:] = 2
        
        batch_of_indices[im] = indices
        batch_of_deltas[im] = deltas
        batch_of_labels[im] = boxlabels
        
        
    return batch_of_indices, batch_of_deltas, batch_of_labels

def indices_deltas_labels(batch_of_bboxes, anchors, batchlen, train_set_size = 20, mode = 'pixelwise'):
    """
    This function generates indices, deltas, and labels for a batch of bounding boxes and anchors for
    use in training a region proposal network.
    
    @param batch_of_bboxes: A list of bounding boxes for each image in the batch. Each bounding box is
    represented as a list of four values: [x, y, width, height]
    
    @param anchors: A list of anchor boxes, which are pre-defined bounding boxes of different sizes and
    aspect ratios that are used to generate region proposals
    
    @param batchlen: The number of images in the batch
    
    @param train_set_size: The number of samples to be selected from the foreground and background boxes
    for each image in the batch, defaults to 20 (optional)
    
    @param mode: The mode parameter specifies whether to use "pixelwise" or "exponential" method for
    calculating deltas, defaults to pixelwise (optional)
    
    @return: a tuple of three arrays: batch_of_indices, batch_of_deltas, and batch_of_labels.
    """
    
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
