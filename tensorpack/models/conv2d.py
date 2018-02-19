#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py


import tensorflow as tf
from .common import layer_register, VariableHolder
from ..tfutils.common import get_tf_version_number
from ..utils.argtools import shape2d, shape4d, get_data_format
from .tflayer import rename_get_variable, convert_to_tflayer_args

__all__ = ['Conv2D', 'Deconv2D']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })

def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1,
        pool3d=False):

    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    if pool3d:
        num_groups = filters
        fs = shape2d(kernel_size)[0]
        filters = filters * fs*fs
        if data_format == 'channels_first':
            channels_first = True
        else:
            channels_first = False


    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            #print ret.name
            #tf.add_to_collection('checkpoints', ret.name)
            ret = tf.identity(ret, name='output')
            #print ret.name, '\n\n\n\n\n\n'

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv now!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == (1, 1) or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        inputs = tf.split(inputs, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)
        if activation is None:
            activation = tf.identity
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b


    if pool3d:
        print("\nTransforming conv layer output feature maps by applying 3D pooling")
        print("Taking {:d} feature maps {}".format(filters, ret.shape))
        import numpy as np

        if channels_first:  # channels first format
            def build_mask(fs, dim, batch_size, debug=False):
                mask = np.zeros((fs * fs, dim, dim))
                for i in range(fs):
                    for j in range(fs):
                        mask[fs * i + j, i::fs, j::fs] = 1

                mask = tf.constant(mask, dtype=tf.float32)
                print("Mask shape: {}".format(mask.get_shape()))
                mask = tf.tile(mask, tf.stack([num_groups, 1, 1]))
                # print("groups_mask shape: {}".format(mask.get_shape()))
                mask = tf.expand_dims(mask, axis=0)
                # print("groups_mask_expanded shape: {}".format(mask.get_shape()))
                mask = tf.tile(mask, tf.stack([batch_size, 1, 1, 1]))
                print("batch_mask shape: {}".format(mask.get_shape()))

                return mask

            dim = ret.shape[-1]
            batch_size = tf.shape(ret)[0]   # evaluates to None during graph construction
            mask = build_mask(fs, dim, batch_size)
            ret = mask * ret
            ret = tf.reshape(ret, (batch_size, num_groups, fs*fs, dim, dim))
            ret = tf.reduce_sum(ret, axis=2)
            print("Outputting {} feature maps {}\n".format(ret.shape[1], ret.shape))

        else:  # channels last format
            def build_mask(fs, dim, batch_size):
                mask = np.zeros((dim, dim, fs * fs))
                for i in range(fs):
                    for j in range(fs):
                        mask[i::fs, j::fs, fs * i + j] = 1

                mask_T = tf.constant(mask, dtype=tf.float32)
                print("Mask shape: {}".format(mask_T.get_shape()))
                groups_mask = tf.tile(mask_T, tf.stack([1, 1, num_groups]))
                # print("groups_mask shape: {}".format(groups_mask.get_shape()))
                groups_mask_expanded = tf.expand_dims(groups_mask, axis=0)
                # print("groups_mask_expanded shape: {}".format(groups_mask_expanded.get_shape()))
                batch_mask = tf.tile(groups_mask_expanded, tf.stack([batch_size, 1, 1, 1]))
                print("batch_mask shape: {}".format(batch_mask.get_shape()))

                return batch_mask

            dim = ret.shape[1]
            batch_size = tf.shape(ret)[0]   # evaluates to None during graph construction
            mask = build_mask(fs, dim, batch_size)
            ret = mask * ret
            ret = tf.reshape(ret, (batch_size, dim, dim, num_groups, fs * fs))
            ret = tf.reduce_sum(ret, axis=-1)
            print("Outputting {} feature maps {}\n".format(ret.shape[3], ret.shape))

    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Deconv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')
