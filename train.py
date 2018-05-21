from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl
import utils


# ****************************************************************************
# *                                   param                                  *
# ****************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--z_dim', dest='z_dim', type=int, default=100, help='dimension of latent')
parser.add_argument('--divergence', dest='divergence', default='Jensen-Shannon', help='divergence',
                    choices=['Kullback-Leibler', 'Reverse-KL', 'Pearson-X2', 'Squared-Hellinger', 'Jensen-Shannon', 'GAN'])
parser.add_argument('--tricky_G', dest='tricky_G', action='store_true', help='use tricky G loss or not')
parser.add_argument('--dataset', dest='dataset_name', default='mnist', choices=['mnist', 'celeba'], help='dataset')

args = parser.parse_args()

epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
z_dim = args.z_dim

divergence = args.divergence
tricky_G = args.tricky_G
dataset_name = args.dataset_name
print(tricky_G)
experiment_name = '%s_%s_%s' % (dataset_name, divergence, 'trickyG' if tricky_G else 'normalG')

# dataset and models
Dataset, models = utils.get_dataset_models(dataset_name)
dataset = Dataset(batch_size=batch_size)
G = models['G']
D = models['D']
activation_fn, conjugate_fn = utils.get_divengence_funcs(divergence)


# ****************************************************************************
# *                                   graph                                  *
# ****************************************************************************

# inputs
real = tf.placeholder(tf.float32, [None, 28, 28, 1])
z = tf.placeholder(tf.float32, [None, z_dim])

# generate
fake = G(z)

# dicriminate
r_output = D(real)
f_output = D(fake)

# losses
d_r_loss = -tf.reduce_mean(activation_fn(r_output))
d_f_loss = tf.reduce_mean(conjugate_fn(activation_fn(f_output)))
d_loss = d_r_loss + d_f_loss
if tricky_G:
    g_loss = -tf.reduce_mean(activation_fn(f_output))
else:
    g_loss = -d_f_loss

# otpims
d_var = tl.trainable_variables('D')
g_var = tl.trainable_variables('G')
d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

# summaries
d_summary = tl.summary({d_r_loss: 'd_r_loss',
                        d_f_loss: 'd_f_loss',
                        -d_loss: '%s_diverngence' % divergence}, scope='D')
g_summary = tl.summary({g_loss: 'g_loss'}, scope='G')

# sample
f_sample = G(z, is_training=False)


# ****************************************************************************
# *                                   train                                  *
# ****************************************************************************

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    it = -1
    it_per_epoch = len(dataset) // batch_size
    for ep in range(epoch):
        dataset.reset()
        for batch in dataset:
            it += 1
            it_in_epoch = it % it_per_epoch + 1

            # batch data
            real_ipt = batch['img']
            z_ipt = np.random.normal(size=[batch_size, z_dim])

            # train D
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, it_in_epoch, it_per_epoch))

            # sample
            if (it + 1) % 1000 == 0:
                f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(f_sample_opt), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, it_in_epoch, it_per_epoch))

        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('Model is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()
