"""
    legacy: copied from colab tested pybook at link: https://colab.research.google.com/drive/1N-elgwGxUJPjucsWd6EFMD8NwnbOu7k5
"""
import numpy as np
import tensorflow as tf
import keras

import sys
import os
import random
import datetime
import re
import math
import logging

import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from keras import backend as K

sys.path.append('.')
sys.path.append('..')

import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize


# Root directory of the project
ROOT_DIR = os.getcwd()
print (ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print (MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
print (IMAGE_DIR)

from samples.coco import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
# model = modellib.MaskRCNN_test(mode="inference", model_dir=MODEL_DIR, config=config)
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

for i, lr in enumerate(model.keras_model.layers, 0):
    print("%d - %s -- %s -- %s -- %s" % (i, lr,lr.name,lr.input_shape, lr.output_shape) )

# ref: http://sujitpal.blogspot.com/2017/10/debugging-keras-networks.html
def get_outputs(inputs, model):
    layer_01_fn = K.function([model.layers[0].input, K.learning_phase()], 
                             [model.layers[1].output]) 
    layer_23_fn = K.function([model.layers[2].input, K.learning_phase()],
                             [model.layers[3].output])
    layer_44_fn = K.function([model.layers[4].input, K.learning_phase()],
                             [model.layers[4].output])
    layer_1_out = layer_01_fn([inputs, 1])[0]
    layer_3_out = layer_23_fn([layer_1_out, 1])[0]
    layer_4_out = layer_44_fn([layer_3_out, 1])[0]
    return layer_1_out, layer_3_out, layer_4_out


# Load weights trained on MS-COCO
print(COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
# results = model.detect([image], verbose=1)

out_1, out_3, out_4 = get_outputs([image], model.keras_model)
print("out_1 {}", (out_1))
print("out_3 {}", (out_3))
print("out_4 {}", (out_4)) 
