#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-visualizations.py

import os
import argparse

"""
MNIST ConvNet example with weights/activations visualization.
"""


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.common import get_op_or_tensor_by_name, get_op_tensor_name
import tensorflow as tf

IMAGE_SIZE = 28


def visualize_filters(filters):
    with tf.name_scope('visualizations'):
        for i,f in enumerate(filters):
            visualize_conv_weights(f, 'conv'+str(i))


def visualize_acts(acts):
    with tf.name_scope('visualizations'):
        for i,f in enumerate(acts):
            visualize_conv_activations(f, 'acts'+str(i))


def visualize_input(image):
    with tf.name_scope('visualizations'):
        tf.summary.image('input', (image + 1.0) * 128., 3)



def visualize_conv_weights(filters, name):
    """Visualize use weights in convolution filters.

    Args:
        filters: tensor containing the weights [H,W,Cin,Cout]
        name: label for tensorboard

    Returns:
        image of all weight
    """
    with tf.name_scope('visualize_w_' + name):
        filters = tf.transpose(filters, (3, 2, 0, 1))   # [h, w, cin, cout] -> [cout, cin, h, w]
        filters = tf.unstack(filters)                   # --> cout * [cin, h, w]
        filters = tf.concat(filters, 1)                 # --> [cin, cout * h, w]
        filters = tf.unstack(filters)                   # --> cin * [cout * h, w]
        filters = tf.concat(filters, 1)                 # --> [cout * h, cin * w]
        filters = tf.expand_dims(filters, 0)
        filters = tf.expand_dims(filters, -1, name='filters')

        #print filters.name, '\n\n\n\n'
        #print get_op_tensor_name(filters.name), '\n\n\n\n'
        #print tf.get_default_graph().get_tensor_by_name("visualizations/visualize_w_conv0/filters:0"), '\n\n\n\n'

        #a = get_op_or_tensor_by_name('visualizations/visualize_w_conv0/filters:0')

    #tf.summary.image('visualize_w_' + name, a)


def visualize_conv_activations(activation, name):
    """Visualize activations for convolution layers.

    Remarks:
        This tries to place all activations into a square.

    Args:
        activation: tensor with the activation [B,H,W,C]
        name: label for tensorboard

    Returns:
        image of almost all activations
    """
    import math
    with tf.name_scope('visualize_act_' + name):
        _, h, w, c = activation.get_shape().as_list()
        rows = []
        c_per_row = int(math.sqrt(c))
        for y in range(0, c - c_per_row, c_per_row):
            row = activation[:, :, :, y:y + c_per_row]  # [?, H, W, 32] --> [?, H, W, 5]
            cols = tf.unstack(row, axis=3)              # [?, H, W, 5] --> 5 * [?, H, W]
            row = tf.concat(cols, 1)
            rows.append(row)

        viz = tf.concat(rows, 2)
    tf.summary.image('visualize_act_' + name, tf.expand_dims(viz, -1))


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):

        image, label = inputs
        self.image = image.name
        image = tf.expand_dims(image * 2 - 1, 3)

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
            c0 = Conv2D('conv0', image)
            p0 = MaxPooling('pool0', c0, 2)
            c1 = Conv2D('conv1', p0)
            c2 = Conv2D('conv2', c1)
            p1 = MaxPooling('pool1', c2, 2)
            c3 = Conv2D('conv3', p1)
            fc1 = FullyConnected('fc0', c3, 512, nl=tf.nn.relu)
            fc1 = Dropout('dropout', fc1, 0.5)
            logits = FullyConnected('fc1', fc1, out_dim=10, nl=tf.identity)

        with tf.name_scope('visualizations'):
            visualize_conv_weights(c0.variables.W, 'conv0')

        #self.filters = [var.name for var in [c0.variables.W, c1.variables.W, c2.variables.W, c3.variables.W]]
        #self.acts = [var.name for var in [c0, c1, c2, c3]]

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(logits, label, 1)), name='accuracy')

        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, self.cost, accuracy)

        summary.add_param_summary(('.*/W', ['histogram', 'rms']))


    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test


def get_config():

    logger.auto_set_dir()
    dataset_train, dataset_test = get_data()
    filters = ['conv0/W:0', 'conv1/W:0', 'conv2/W:0', 'conv3/W:0']

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            PeriodicTrigger(RunOp(tf.summary.image('visualize_w_' + 'conv0', get_op_or_tensor_by_name(
                'visualizations/visualize_w_conv0/filters:0')))),
            #PeriodicRunHooks(ProcessTensors([filters], visualize_filters), every_k_steps=dataset_train.size()),
            ModelSaver(),
            InferenceRunner(
                dataset_test, ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=dataset_train.size(),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SimpleTrainer())
