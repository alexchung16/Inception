#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v3.py
# @ Description :
# @ Author alexchung
# @ Time 1/11/2019 PM 15:27

# Conv2d_1a_3x3
# Conv2d_2a_3x3
# Conv2d_2b_3x3
# MaxPool_3a_3x3
# Conv2d_3b_1x1
# Conv2d_4a_3x3
# MaxPool_5a_3x3
# Mixed_5b
# Mixed_5c
# Mixed_5d
# Mixed_6a
# Mixed_6b
# Mixed_6c
# Mixed_6d
# Mixed_6e
# Mixed_7a
# Mixed_7b
# Mixed_7c




import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class InceptionV3():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_steps, decay_rate, learning_rate,
                 keep_prob=0.8, global_pool=False, spacial_squeeze=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.global_pool = global_pool
        self.spacial_squeeze = spacial_squeeze
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder (tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]], name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder (tf.float32, shape=[None, self.num_classes], name="class_label")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")

        # logits
        self.logits =  self.inference(self.raw_input_data, scope='InceptionV3')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.evaluate_accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size


    def inference(self, inputs, scope='InceptionV3'):
        """
        Inception V3 net structure
        :param inputs:
        :param scope:
        :return:
        """
        pass

    def inception_v3(self):
        pass


    def inception_v3_base(self):
        pass

    def inception_module_v3_a(self, input_op, filters_list, scope):
        """
        inception v3 module a
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch 0
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[0], strides=1)
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[1], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', kernel_size=[5, 5],
                                       filters=filters_list[2], strides=1)
                # branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                #                        filters=filters_list[2], strides=1)
                # branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0c_3x3', kernel_size=[3, 3],
                #                        filters=filters_list[3], strides=1)
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[3], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                                       filters=filters_list[4], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0c_3x3', kernel_size=[3, 3],
                                       filters=filters_list[5], strides=1)
            # branch 3
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = avgPool2dLayer(input_op=input_op, scope='AvgPool_3x3', kernel_size=[3, 3], strides=1)
                branch_3 = conv2dLayer(input_op=branch_1, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[6], strides=1)
            net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)


    def inception_module_v3_b(self):
        pass

    def inception_module_v3_c(self):
        pass

    def inception_module_v3_reduce(self):
        pass


def conv2dLayer(input_op, scope, filters, kernel_size=None, strides=None, use_bias=False, padding='SAME', parameter=None):
    """
    convolution operation
    :param input_op:
    :param scope:
    :param filters:
    :param kernel_size:
    :param strides:
    :param use_bias:
    :param padding:
    :param parameter:
    :return:
    """
    if kernel_size is None:
        kernel_size = [3, 3]
    if strides is None:
        strides = 1
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.name_scope(scope) as scope:
        filter = getConvFilter(filter_shape=[kernel_size[0], kernel_size[1], features, filters])

        outputs = tf.nn.conv2d(input=input_op, filter=filter, strides=[1, strides, strides, 1], padding=padding)

        if use_bias:
            biases = getBias(bias_shape=[filters])
            outputs = tf.nn.bias_add(value=outputs, bias=biases)
        #     parameter += [filter, biases]
        # else:
        #     parameter += [filters]

        return tf.nn.relu(outputs)


def fcLayer(input_op, scope, num_outputs, parameter=None):
    """
     full connect operation
    :param input_op:
    :param scope:
    :param num_outputs:
    :param parameter:
    :return:
    """
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.name_scope(scope) as scope:
        weights = getFCWeight(weight_shape=[features, num_outputs])
        biases = getBias(bias_shape=[num_outputs])
        # parameter += [weights, biases]
        return tf.nn.relu_layer(x=input_op, weights=weights, biases=biases)


def maxPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='SAME'):
    """
     max pooling layer
    :param input_op:
    :param scope:
    :param kernel_size:
    :param strides_size:
    :param padding:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        if kernel_size is None:
            ksize = [1, 2, 2, 1]
        else:
            ksize = [1, kernel_size[0], kernel_size[1], 1]
        if strides is None:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, strides, strides, 1]
        return tf.nn.max_pool2d(input=input_op, ksize=ksize, strides=strides, padding=padding, name='MaxPool')


def avgPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='SAME'):
    """
    average_pool pooling layer
    :param input_op:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        if kernel_size is None:
            ksize = [1, 2, 2, 1]
        else:
            ksize = [1, kernel_size[0], kernel_size[1], 1]
        if strides is None:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, strides, strides, 1]
        return tf.nn.avg_pool2d(value=input_op, ksize=ksize, strides=strides, padding=padding, name='AvgPool')


def flatten(input_op, scope):
    """
    flatten layer
    :param input_op:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        shape = input_op.get_shape().as_list()
        out_dim = 1
        for d in shape[1:]:
            out_dim *= d
        return tf.reshape(tensor=input_op, shape=[-1, out_dim], name='Flatten')


def dropoutLayer(input_op, scope, keep_prob):
    """
    dropout regularization layer
    :param inpu_op:
    :param name:
    :param keep_prob:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        return tf.nn.dropout(input_op, rate=1-keep_prob, name='Dropout')


def softmaxLayer(input_op, scope):
    """
    softmax layer
    :param logits:
    :param name:
    :param n_class:
    :return:
    """
    with tf.compat.v1.variable_scope(scope):
        return tf.nn.softmax(logits=input_op, name='Softmax')


def getConvFilter(filter_shape):
    """
    convolution layer filter
    :param filter_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=filter_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
                       trainable=True, name='Filter')


def getFCWeight(weight_shape):
    """
    full connect layer weight
    :param weight_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=weight_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
                       trainable=True, name='Weight')


def getBias(bias_shape):
    """
    get bias
    :param bias_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.constant(value=0.0, shape=bias_shape, dtype=tf.float32),
                       trainable=True, name='Bias')