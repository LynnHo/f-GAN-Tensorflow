from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)


def G(z, dim=64, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu)

    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        return img


def D(img, dim=64, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit
