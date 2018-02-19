#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

#gradient checkpointing to save memory (expect slowdown):
import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
from tensorflow.python.ops import gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):

    tensors = ['tower0/conv0/Relu:0', 'tower0/group0/block0/conv1/Relu:0', 'tower0/group0/block0/conv2/bn/FusedBatchNorm:0',
     'tower0/group0/block1/conv1/Relu:0', 'tower0/group0/block1/conv2/bn/FusedBatchNorm:0',
     'tower0/group1/block0/conv1/Relu:0', 'tower0/group1/block0/conv2/bn/FusedBatchNorm:0',
     'tower0/group1/block0/convshortcut/bn/FusedBatchNorm:0', 'tower0/group1/block1/conv1/Relu:0',
     'tower0/group1/block1/conv2/bn/FusedBatchNorm:0', 'tower0/group2/block0/conv1/Relu:0',
     'tower0/group2/block0/conv2/bn/FusedBatchNorm:0', 'tower0/group2/block0/convshortcut/bn/FusedBatchNorm:0',
     'tower0/group2/block1/conv1/Relu:0', 'tower0/group2/block1/conv2/bn/FusedBatchNorm:0',
     'tower0/group3/block0/conv1/Relu:0', 'tower0/group3/block0/conv2/bn/FusedBatchNorm:0',
     'tower0/group3/block0/convshortcut/bn/FusedBatchNorm:0', 'tower0/group3/block1/conv1/Relu:0',
     'tower0/group3/block1/conv2/bn/FusedBatchNorm:0']

    tensors = ['tower0/conv0/Relu:0', 'tower0/group0/block0/conv1/Relu:0']

    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints=tensors, **kwargs)
#gradients.__dict__["gradients"] = memory_saving_gradients.gradients_memory
gradients.__dict__["gradients"] = gradients_memory


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format == 'NCHW' else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    scale1 = 2#0.75
    scale2 = 2#0.5
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'), pool3d=False):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', int(64*scale1), 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, int(64*scale1), num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, int(128*scale2), num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, int(256*scale2), num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, int(512*scale2), num_blocks[3], 2)
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 1000, nl=tf.identity)())

    print '\n\n\n\n'
    print tf.get_collection('checkpoints')
    for var in tf.get_collection('checkpoints'):
        print var
    print '\n\n\n\n'

    return logits
