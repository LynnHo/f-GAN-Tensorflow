from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import pylib
import tensorflow as tf
import tflib as tl


def get_divergence_funcs(divergence):
    if divergence == 'Kullback-Leibler':
        def activation_fn(v): return v

        def conjugate_fn(t): return tf.exp(t - 1)

    elif divergence == 'Reverse-KL':
        def activation_fn(v): return -tf.exp(-v)

        def conjugate_fn(t): return -1 - tf.log(-t)

    elif divergence == 'Pearson-X2':
        def activation_fn(v): return v

        def conjugate_fn(t): return 0.25 * t * t + t

    elif divergence == 'Squared-Hellinger':
        def activation_fn(v): return 1 - tf.exp(-v)

        def conjugate_fn(t): return t / (1 - t)

    elif divergence == 'Jensen-Shannon':
        def activation_fn(v): return tf.log(2.0) - tf.log(1 + tf.exp(-v))

        def conjugate_fn(t): return -tf.log(2 - tf.exp(t))

    elif divergence == 'GAN':
        def activation_fn(v): return -tf.log(1 + tf.exp(-v))

        def conjugate_fn(t): return -tf.log(1 - tf.exp(t))

    return activation_fn, conjugate_fn


def get_dataset_models(dataset_name):
    if dataset_name == 'mnist':
        import models
        pylib.mkdir('./data/mnist')
        Dataset = partial(tl.Mnist, data_dir='./data/mnist', repeat=1)
        return Dataset, {'D': models.D, 'G': models.G}

    elif dataset_name == 'celeba':
        import models_64x64
        raise NotImplementedError
