import matplotlib.pyplot as plt
import numpy as np
from utils import get_proposals, draw_bbox

def visualize_rpn_result(image_batch, pred_scores, pred_deltas, anchors, proposal_count = 20, mode = 'pixelwise'):
    """
    The function visualizes the region proposal network (RPN) results by plotting the predicted bounding
    boxes on the input images.
    
    @param image_batch: A batch of images in the form of a numpy array with shape (batch_size, height,
    width, channels)
    
    @param pred_scores: The predicted scores for each anchor box, which indicate the likelihood of the
    box containing an object
    
    @param pred_deltas: pred_deltas is a tensor containing the predicted bounding box deltas for each
    anchor in the input image. These deltas are used to adjust the coordinates of the anchor boxes to
    generate the final predicted bounding boxes
    
    @param anchors: Anchors are pre-defined bounding boxes of different sizes and aspect ratios that are
    used as reference points for object detection in an image. They are typically generated based on the
    characteristics of the dataset being used for training. In this function, the anchors are used along
    with the predicted scores and deltas to generate
    
    @param proposal_count: proposal_count is the number of proposals to be generated from the predicted
    scores and deltas. These proposals are potential object bounding boxes that the model predicts to
    contain objects in the image, defaults to 20 (optional)
    
    @param mode: The mode parameter specifies the type of visualization to be used. It can be either
    'pixelwise' or 'proposalwise'. If mode is set to 'pixelwise', the function will visualize the
    predicted bounding boxes for each pixel in the image. If mode is set to 'proposalwise', the
    function, defaults to pixelwise (optional)
    """
    
    proposals = get_proposals(pred_scores,pred_deltas,anchors,proposal_count)
    for num, image in enumerate(image_batch):
        plt.figure()
        plt.imshow(image, cmap= 'gray')
        for pred_bbox in proposals[num]:
            plt.plot(*draw_bbox(pred_bbox), color = 'red', linewidth = 0.5, alpha = 1)
    plt.show()
    
def visualize_ch_results(image_batch, predicted_label_batch,
                         predicted_boxes_batch, predicted_scores_batch,
                         classdict, batchlen):
    """
    This function visualizes the results of a convolutional neural network's predictions on a batch of
    images, including the predicted labels, boxes, and scores.
    
    @param image_batch: A batch of images to visualize the results on
    
    @param predicted_label_batch: The predicted labels for each image in the batch
    
    @param predicted_boxes_batch: A batch of predicted bounding boxes for each image in the input batch
    
    @param predicted_scores_batch: The predicted confidence scores for each bounding box in the image
    batch
    
    @param classdict: A dictionary that maps class labels to their corresponding names or descriptions
    
    @param batchlen: The number of images in the batch
    """
    
    for i in range(batchlen):
        plt.figure()
        plt.show(image_batch[i], cmap='gray')
        for num, box in enumerate(predicted_boxes_batch[i]):
            if np.all(np.equal(box, 0)):
                continue
            else:
                plt.plot(*draw_bbox(box), linewidth= 2, alpha= 1, color= 'pink')
                plt.text(box[1] + 50, box[0] - 5, predicted_scores_batch[i][num], color = 'pink', fontsize = 12)
                plt.text(box[1], box[0] - 5, classdict[predicted_label_batch[i][num]], color = 'pink', fontsize = 12)

def visualize_results(image_batch,predicted_mask_batch,predicted_label_batch,predicted_boxes_batch,predicted_scores_batch,classdict,batchlen,gt_mask_batch=None):
    """
    This function visualizes the results of a model's predictions on a batch of images, including
    predicted masks, labels, boxes, and scores, using a dictionary of class labels and optionally
    comparing to ground truth masks.
    
    @param image_batch: A batch of input images to the model
    
    @param predicted_mask_batch: A batch of predicted masks for the input images. Each mask is a 2D
    array of the same size as the input image, with pixel values indicating the probability of the
    corresponding pixel belonging to the object of interest
    
    @param predicted_label_batch: A batch of predicted labels for the input images. These labels are
    predicted by a machine learning model and represent the class of the object present in the image
    
    @param predicted_boxes_batch: A batch of predicted bounding boxes for each image in the input batch.
    Each bounding box is represented as a list of four values: [xmin, ymin, xmax, ymax]
    
    @param predicted_scores_batch: This parameter is a batch of predicted scores for each object
    detected in the input image batch. It is a numpy array of shape (batchlen, num_objects) where
    batchlen is the number of images in the batch and num_objects is the maximum number of objects that
    can be detected in an image
    
    @param classdict: A dictionary that maps class indices to class names. For example, {0:
    'background', 1: 'person', 2: 'car'}
    
    @param batchlen: The number of images in the batch
    
    @param gt_mask_batch: The ground truth mask batch for the images in the input batch. It is an
    optional parameter and can be set to None if not available
    """
    
    if batchlen > 1:
        if gt_mask_batch is None:
            f, axarr = plt.subplots(batchlen, 2)
        else:
            f, axarr = plt.subplots(batchlen, 3)
        for i in range(batchlen):
            axarr[i,0].imshow(image_batch[i], cmap= 'gray')
            axarr[i,1].imshow(predicted_mask_batch[i], cmap= 'gray')
            for num, box in enumerate(predicted_boxes_batch[i]):
                if np.all(np.equal(box, 0)):
                    continue
                else:
                    axarr[i,0].plot(*draw_bbox(box), linewidth= 2, alpha= 1, color= 'pink')
                    axarr[i,1].text(box[1] + 70, box[0] - 5, predicted_scores_batch[i][num], color= 'pink', fontsize= 12)
                    axarr[i,1].text(box[1], box[0] - 5, classdict[predicted_label_batch[i][num]], color = 'pink', fontsize = 12)
                    axarr[i,1].plot(*draw_bbox(box), linewidth = 2, alpha = 1, color = 'pink')
                
            if gt_mask_batch is not None:
                axarr[i,2].imshow(gt_mask_batch[i], cmap='gray')
            
    
    else:
        if gt_mask_batch is None:
            f, axarr = plt.subplots(1,2)
        else:
            f, axarr = plt.subplots(1,3)
        
        axarr[0].imshow(image_batch[0], cmap='gray')
        axarr[1].imshow(predicted_mask_batch[0], cmap='gray')
        
        for num, box in enumerate(predicted_boxes_batch[0]):
            if np.all(np.equal(box,0)):
                continue
            else:
                axarr[0].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                axarr[1].text(box[1]+50,box[0]-5,predicted_scores_batch[0][num],color='pink',fontsize=12)
                axarr[1].text(box[1],box[0]-5,classdict[predicted_label_batch[0][num]],color='pink',fontsize=12)
                axarr[1].plot(*draw_bbox(box),linewidth=2, alpha=1, color='pink')
                
        if gt_mask_batch is None:
            axarr[2].imshow(gt_mask_batch[0], cmap= 'gray')