"""
    legacy: copied from colab tested pybook at link: https://colab.research.google.com/drive/1N-elgwGxUJPjucsWd6EFMD8NwnbOu7k5

1) after using upgrade_to_tf2

Traceback (most recent call last):
  File "samples\test_sl.py", line 62, in <module>
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
  File ".\mrcnn\model.py", line 1852, in __init__
    self.keras_model = self.build(mode=mode, config=config)
  File ".\mrcnn\model.py", line 1871, in build
    shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
  File "c:\PYTHON3.6\lib\site-packages\keras\engine\topology.py", line 1457, in Input
    input_tensor=tensor)
  File "c:\PYTHON3.6\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "c:\PYTHON3.6\lib\site-packages\keras\engine\topology.py", line 1366, in __init__
    name=self.name)
  File "c:\PYTHON3.6\lib\site-packages\keras\backend\tensorflow_backend.py", line 507, in placeholder
    x = tf.placeholder(dtype, shape=shape, name=name)
AttributeError: module 'tensorflow' has no attribute 'placeholder'

solution: perhaps keras version
ref: https://github.com/CyberZHG/keras-bert/issues/24



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

import cv2

sys.path.append('.')
sys.path.append('..')

import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

print('eagerly?',tf.executing_eagerly())

# Root directory of the project
ROOT_DIR = os.getcwd()
print ('ROOT_DIR = ',ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print (MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print('COCO_MODEL_PATH not exists.')
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
print ('IMAGE_DIR = ', IMAGE_DIR)

from samples.coco import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image_name = os.path.join(IMAGE_DIR, random.choice(file_names))
print('image name=', image_name)
image = skimage.io.imread(image_name)
assert image is not None
print('image shape=', image.shape)
image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('read image', image_)
cv2.waitKey()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

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

# Run detection
results = model.detect([image], verbose=1)

exit()

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])