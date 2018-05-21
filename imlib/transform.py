from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imlib.dtype import *
import numpy as np
import scipy.misc


def rgb2gray(images):
    if images.ndim == 4 or images.ndim == 3:
        assert images.shape[-1] == 3, 'Channel size should be 3!'
    else:
        raise Exception('Wrong dimensions!')

    return (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114).astype(images.dtype)


def gray2rgb(images):
    assert images.ndim == 2 or images.ndim == 3, 'Wrong dimensions!'
    rgb_imgs = np.zeros(images.shape + (3,), dtype=images.dtype)
    rgb_imgs[..., 0] = images
    rgb_imgs[..., 1] = images
    rgb_imgs[..., 2] = images
    return rgb_imgs


def imresize(image, size, interp='bilinear'):
    """Resize an [-1.0, 1.0] image.

    Args:
        size : int, float or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos',
            'bilinear', 'bicubic' or 'cubic').
    """
    # scipy.misc.imresize should deal with uint8 image, or it would cause some
    # problem (scale the image to [0, 255])
    return (scipy.misc.imresize(im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)


def resize_images(images, size, interp='bilinear'):
    """Resize batch [-1.0, 1.0] images of shape (N * H * W (* 3)).

    Args:
        size : int, float or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos',
            'bilinear', 'bicubic' or 'cubic').
    """
    rs_imgs = []
    for img in images:
        rs_imgs.append(imresize(img, size, interp))
    return np.array(rs_imgs)


def immerge(images, n_row=None, n_col=None, padding=0, pad_value=0):
    """Merge images into an image with (n_row * h) * (n_col * w).

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    n = images.shape[0]
    if n_row:
        n_row = max(min(n_row, n), 1)
        n_col = int(n - 0.5) // n_row + 1
    elif n_col:
        n_col = max(min(n_col, n), 1)
        n_row = int(n - 0.5) // n_col + 1
    else:
        n_row = int(n ** 0.5)
        n_col = int(n - 0.5) // n_row + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_row + padding * (n_row - 1),
             w * n_col + padding * (n_col - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_col
        j = idx // n_col
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img
