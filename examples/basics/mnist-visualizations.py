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

class VisualizeTensors(Callback):
    """
    Prepare and export tensors (filters, activations,
    and input images) to TensorBoard for visualization
    """

    def __init__(self, filters, acts):
        """
        Args:
            filters: list of strings, the names of the conv. layer filters to visualize.
            acts: list of strings, the names of the conv. layer activations to visualize.
        """
        filters = [get_op_tensor_name(n)[1] for n in filters]
        acts = [get_op_tensor_name(n)[1] for n in acts]
        self.image = get_op_tensor_name('input_image')[1]
        self.filter_names = filters
        self.act_names = acts

    def _setup_graph(self):
        filters = self.get_tensors_maybe_in_tower(self.filter_names)
        acts = self.get_tensors_maybe_in_tower(self.act_names)
        image = self.get_tensors_maybe_in_tower([self.image])

        with tf.name_scope('visualizations'):
            tf.summary.image('input', (image[0] + 1.0) * 128., 3)
            for i, (f, a) in enumerate(zip(filters, acts)):
                self.visualize_conv_weights(f, 'conv' + str(i))
                self.visualize_conv_activations(a, 'acts' + str(i))

    def visualize_conv_weights(self, filters, name):
        """Visualize weights in convolution filters.

        Args:
            filters: tensor containing the weights [H,W,Cin,Cout]
            name: label for tensorboard

        Returns:
            image of all weight
        """
        with tf.name_scope('visualize_w_' + name):
            filters = tf.transpose(filters, (3, 2, 0, 1))  # [h, w, cin, cout] -> [cout, cin, h, w]
            filters = tf.unstack(filters)  # --> cout * [cin, h, w]
            filters = tf.concat(filters, 1)  # --> [cin, cout * h, w]
            filters = tf.unstack(filters)  # --> cin * [cout * h, w]
            filters = tf.concat(filters, 1)  # --> [cout * h, cin * w]
            filters = tf.expand_dims(filters, 0)
            filters = tf.expand_dims(filters, -1)

        tf.summary.image(name, filters)


    def visualize_conv_activations(self, activation, name):
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
                cols = tf.unstack(row, axis=3)  # [?, H, W, 5] --> 5 * [?, H, W]
                row = tf.concat(cols, 1)
                rows.append(row)

            viz = tf.concat(rows, 2)
        tf.summary.image(name, tf.expand_dims(viz, -1))


    def _trigger(self):
        pass


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):

        image, label = inputs
        image = tf.expand_dims(image * 2 - 1, 3, name='input_image')

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
    acts = ['conv0/output:0', 'conv1/output:0', 'conv2/output:0', 'conv3/output:0']

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            VisualizeTensors(filters, acts),
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
