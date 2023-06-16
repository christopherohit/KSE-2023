import numpy as np

def calculate_pixelwise_deltas(bbox, anchors, numof):
    """
    This function calculates the predicted dx, dy, dw, and dh for each anchor based on the given
    bounding box and anchor parameters.
    
    @param bbox: The bounding box coordinates of an object in the format [xmin, ymin, xmax, ymax]
    
    @param anchors: A numpy array of shape (numof, 4) containing the coordinates of the anchor boxes.
    Each row represents an anchor box and the four columns represent the x-coordinate of the top-left
    corner, y-coordinate of the top-left corner, x-coordinate of the bottom-right corner, and
    y-coordinate of
    
    @param numof: The number of anchors
    
    @return: a numpy array of shape (numof, 4) containing the predicted dx, dy, dw, dh values for each
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

def calculate_ious(bbox, anchors):
    """
    This function calculates the intersection over union (IOU) between a bounding box and a set of
    anchor boxes.
    
    @param bbox: The bbox parameter is a list or array containing the coordinates of a bounding box in
    the format [y1, x1, y2, x2], where y1 and x1 are the coordinates of the top-left corner of the box,
    and y2 and x2 are the coordinates
    
    @param anchors: The anchors parameter is a numpy array containing the coordinates of the anchor
    boxes. Each row of the array represents an anchor box and contains four values: the x-coordinate of
    the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right
    corner, and the y-coordinate
    
    @return: an array of Intersection over Union (IoU) values between a bounding box and a set of anchor
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

