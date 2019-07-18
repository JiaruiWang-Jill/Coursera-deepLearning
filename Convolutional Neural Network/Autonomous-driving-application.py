import argparse
import os
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import scipy.io 
import scipy.misc 
import numpy as np 
import pandas as pd 
import PIL
import tensorflow as tf 
from keras import backend as K 
from keras.layers import load_model, Model 
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yol_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	box_scores = box_confidence * box_class_probs
	box_classes = K.argmax(box_scores, axis = -1)
	box_class_scores = K.max(box_scores, axis = -1)

	filtering_mask = box_class_scores >= threshold

	scores = tf.boolean_mask(box_class_scores, filtering_mask)
	boxes = tf.boolean_mask(boxes, filtering_mask)
	classes = tf.boolean_mask(box_classes, filtering_mask)


	return scores, boxes, classes



def iou(box1, box2):
	xi1 = max([box1[0],box2[0]])
    yi1 = max([box1[1],box2[1]])
    xi2 = min([box1[2],box2[2]])
    yi2 = min([box1[3],box2[3]])
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
	box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
	union_area = box1_area + box2_area - inter_area

	iou = inter_area / union_area
	return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
	max_boxes_tensor = K.variable(max_boxes, dtype='int32')
	K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

	nms_indices = tf.image.non_max_suppression(boxes = boxes,scores=scores, max_output_size=max_boxes,iou_threshold=iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)



def yolo_eval(yolo_outputs, iamge_shape=(720., 1280.), max_boxes=10, score_threshold = 0.5):
	box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

	boxes = yolo_boxes_to_corners(box_xy, box_wh)

	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs) 

	boxes = scale_boxes(boxes, iamge_shape)

	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, 10, 0.5)

	return scores, boxs, classes



with tf.Session() as test_a: 
	box_confidence = tf.random_normal([19,19,5,1], mean = 1, stddev = 4, seed=1)
	boxes = tf.random_normal([19,19,5,4], mean = 1, stddev = 4, seed = 1)
	box_class_probs = tf.random_normal([19,19,5,89], mean = 1, stddev =4, seed = 1)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))