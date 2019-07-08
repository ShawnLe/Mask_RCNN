from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.layers as KL

import numpy as np

class dummyClass(KL.Layer):    

    def __init__(self, **kwargs): 
        super(dummyClass, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        # this function is not called in TF 2.0
        print('[compute_output_shape] starts...')
        return (None, None, None)


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

class rpn_model(keras.Model):
    '''
        adapt build_rpn_model to TF2.0 style
    '''
    def __init__(self, anchor_stride, anchors_per_location, depth):
        super(rpn_model, self).__init__()

        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride

    def call(self, input_feature_map):
        outputs = rpn_graph(input_feature_map, self.anchors_per_location, self.anchor_stride)
        return outputs

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

# @tf.function -> no need
def batchnorm_caller(input):
    return BatchNorm()(input)


if __name__ == '__main__':

    print('eagerly?',tf.executing_eagerly())

    # input = KL.Input(shape=(1,2,3))
    dummy = dummyClass()
    print(dummy(tf.constant([1,2,3])))

    rpn = rpn_model(1, len([0.5, 1, 2]), 256)

    rpn_out = rpn(tf.random.normal((1,48,64,3)))
    print('test rpn =\n', rpn_out)
    # print('test rpn shape =\n', rpn_out.shape)

    # print('batch norm test: ',keras.layers.BatchNormalization()(tf.random.normal((1,48,64,3))))

    print('batch norm caller = ', batchnorm_caller(tf.cast(np.random.rand(1,512,512,64), tf.float32)))  # cast np array to tf tensor

    exit()
    
    # print('batch norm test: ', BatchNorm()(tf.random.normal((1,512,512,64)))) # no error
    print('batch norm test: ', BatchNorm()(tf.convert_to_tensor(np.random.rand(1,512,512,64)))) # error 
    print('batch norm test: ', BatchNorm()(tf.convert_to_tensor(np.random.rand(1,512,512,64)))) # no error