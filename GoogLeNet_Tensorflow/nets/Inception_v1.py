#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v1.py
# @ Description : GoogLeNet
# @ Author alexchung
# @ Time 29/10/2019 AM 10:29


import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d


class InceptionV1():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, weight_decay=0.00004,
                 learning_rate=0.01, momentum = 0.9, decay_rate=0.96,  num_samples_per_epoch=None,
                 num_epochs_per_decay=8,keep_prob=0.8, spacial_squeeze=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = int(num_samples_per_epoch * num_epochs_per_decay / batch_size)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.keep_prob = keep_prob
        self.spacial_squeeze = spacial_squeeze

        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 3],
                                                       name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")

        self.keep_prob = tf.compat.v1.placeholder(tf.bool, shape=(), name="keep_prob")

        self.global_step = tf.train.get_or_create_global_step()

        # logits
        self.logits =  self.inference(self.raw_input_data, scope='InceptionV1')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.accuracy = self.get_accuracy(self.logits, self.raw_input_label)


    def inference(self, inputs, scope='InceptionV1'):
       """
       Inception V1 net structure
       :param input_op:
       :param name:
       :return:
       """
       prop =  self.inception_v1(inputs = inputs,
                                 num_classes = self.num_classes,
                                 keep_prob = self.keep_prob,
                                 spatial_squeeze = self.spacial_squeeze,
                                 scope=scope
                                 )
       return prop


    def inception_v1(self, inputs, scope='InceptionV1', num_classes=10, keep_prob=0.8, global_pool=False,
                     spatial_squeeze=True):
        """
        Inception V1 architecture
        :param inputs:
        :param scope:
        :param num_class:
        :param keep_prob:
        :param global_pool:
        :param spatial_squeeze:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV1', values=[inputs]) as scope:
            net = self.inception_v1_base(inputs=inputs, scope=scope)
            with tf.compat.v1.variable_scope('logits'):
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='Global_Pool')
                else:
                    net = avgpool2d(input_op=net, scope='AvgPool_0a_7x7', kernel_size=[7, 7], strides=1, padding='VALID')
                # dropout
                net = dropout(input_op=net, scope='Dropout_0b', keep_prob=keep_prob)
                # conv layer 1*1*num_class
                logits =conv2d(input_op=net, scope='Conv2d_0c_1x1', output_channels=num_classes, kernel_size=[1, 1],
                                     strides=1)
                if spatial_squeeze:
                    logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')
                prop = tf.nn.softmax(input_op=logits, scope='prob')
        return prop


    def inception_v1_base(self, inputs, scope='InceptionV1'):
        """
        Inception V1 base architecture
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV1', values=[inputs]):
            # conv_7*7*64
            net = conv2d(input_op=inputs, scope='Conv2d_1a_7x7', output_channels=64, kernel_size=[7, 7], strides=2)

            # max pool 3*3
            net = maxpool2d(input_op=net, scope='MaxPool_2a_3x3', kernel_size=[3, 3], strides=2)
            # conv 1*1*64
            net = conv2d(input_op=net, scope='Conv2d_2b_1x1', output_channels=64, kernel_size=[1, 1], strides=1)
            # conv 3*3*192
            net = conv2d(input_op=net, scope='Conv2d_2c_3x3', output_channels=192, kernel_size=[3, 3], strides=1)

            # max pool 3*3
            net = maxpool2d(input_op=net, scope='MaxPool_3a_3x3', kernel_size=[3, 3], strides=2)
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_3b', channels_list=[64, 96, 128, 16, 32, 32])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_3c', channels_list=[128, 128, 192, 32, 96, 64])
            # inception module

            # max pool 3*3
            net = maxpool2d(input_op=net, scope='MaxPool_4a_3x3', kernel_size=[3, 3], strides=2)
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_4b', channels_list=[192, 92, 208, 16, 48, 64])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_4c', channels_list=[160, 112, 224, 24, 64, 64])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_4b', channels_list=[128, 128, 256, 24, 64, 64])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_4e', channels_list=[112, 144, 288, 32, 64, 64])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_4f', channels_list=[256, 160, 320, 32, 128, 128])

            # max pool 2*2
            net = maxpool2d(input_op=net, scope='MaxPool_5a_2x2', kernel_size=[2, 2], strides=2)
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_5b', channels_list=[256, 160, 320, 32, 128, 128])
            # inception module
            net = self.inception_module_v1(input_op=net, scope='Mixed_5c', channels_list=[384, 192, 384, 48, 128, 128])

            return net

    def inception_module_v1(self, input_op, channels_list, scope):
        """
        inception module
        :param net:
        :param channels_list:[branch_0_filter,
                        branch_1_filter_0, branch_1_filter_1,
                        branch_2_filter_0, branch_2_filter_1,
                        branch_3_filter_0]
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch0
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = conv2d(input_op=input_op, scope='Conv2d_0a_1x1', output_channels=channels_list[0],
                                       kernel_size=[1, 1], strides=1)
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2d(input_op=input_op, scope='Conv2d_0a_1x1', output_channels=channels_list[1],
                                       kernel_size=[1, 1], strides=1)
                branch_1 = conv2d(input_op=branch_1, scope='Conv2d_0b_3x3', output_channels=channels_list[2],
                                       kernel_size=[3, 3], strides=1)
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = conv2d(input_op=input_op, scope='Conv2d_0a_1x1', output_channels=channels_list[3],
                                       kernel_size=[1, 1], strides=1)
                branch_2 = conv2d(input_op=branch_2, scope='Conv2d_0b_3x3', output_channels=channels_list[4],
                                       kernel_size=[3, 3], strides=1)
            # branch 3
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = maxpool2d(input_op=input_op, scope='MaxPool_0a_3x3', kernel_size=[3, 3], strides=1)
                branch_3 = conv2d(input_op=branch_3, scope='Conv2d_0b_3x3', output_channels=channels_list[5],
                                       kernel_size=[3, 3], strides=1)
            # concat branch
            output_op = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Concat')

        return output_op

    def training(self, learning_rate, global_step, trainable_scope=None):
        """
        According to the paper: use momentum optimizer with 0.9 momentum and
                                fixed learning rate schedule(decrease the learning rate by 4%  every 8 epoch)

        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        trainable_scope = ['']
        if trainable_scope is not None:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]
        else:
            trainable_variable = None

        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum)
        train_op = optimizer.minimize(self.loss, global_step=global_step, var_list=trainable_variable)

        return train_op

    def losses(self, logits, labels, scope='Loss'):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(scope) as sc:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='Entropy_Mean')

    def get_accuracy(self, logits, labels, scope='Evaluate_Batch'):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        with tf.name_scope(scope):
            correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed, keep_prob=1.0):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.keep_prob: keep_prob
        }
        return feed_dict


    def load_weights(self, sess, model_path, custom_scope=None):
        """
        load pre train model
        :param sess:
        :param model_path:
        :param custom_scope:
        :return:
        """
        model_variable = tf.model_variables()
        if custom_scope is None:
            custom_scope = ['InceptionV1/Conv2d_0c_1x1']
        for scope in custom_scope:
            variables = tf.model_variables(scope=scope)
            [model_variable.remove(var) for var in variables]
        saver = tf.train.Saver(var_list=model_variable)
        saver.restore(sess, save_path=model_path)
        print('Successful load pretrain model from {0}'.format(model_path))




def conv2d(input_op, scope, output_channels, kernel_size=None, strides=None, use_bias=True, padding='SAME',
                 fineturn=False, xavier=False):
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
    with tf.compat.v1.variable_scope(scope):
        weights = get_conv_filter(shape=[kernel_size[0], kernel_size[1], features, output_channels],
                               trainable=fineturn, xavier=xavier)

        outputs = tf.nn.conv2d(input=input_op, filter=weights, strides=[1, strides, strides, 1], name=scope,
                               padding=padding)

        if use_bias:
            biases = get_bias(shape=[output_channels], trainable=fineturn)
            outputs = tf.nn.bias_add(value=outputs, bias=biases)

        return tf.nn.relu(outputs)


def fully_connected(input_op, scope, num_outputs, is_activation=True, fineturn=False, xavier=False):
    """
     full connect operation
    :param input_op:
    :param scope:
    :param num_outputs:
    :param parameter:
    :return:
    """
    # get feature num
    shape = input_op.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    with tf.compat.v1.variable_scope(scope):
        flat_data = tf.reshape(tensor=input_op, shape=[-1, size], name='Flatten')

        weights =get_fc_weight(shape=[size, num_outputs], trainable=fineturn, xavier=xavier)
        biases = get_bias(shape=[num_outputs], trainable=fineturn)

        if is_activation:
             return tf.nn.relu_layer(x=flat_data, weights=weights, biases=biases)
        else:
            return tf.nn.bias_add(value=tf.matmul(flat_data, weights), bias=biases)

def maxpool2d(input_op, scope, kernel_size=None, strides=None, padding='VALID'):
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


def avgpool2d(self, input_op, scope, kernel_size=None, strides=None, padding='VALID'):
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


def dropout(self, input_op, scope, keep_prob):
    """
    dropout regularization layer
    :param inpu_op:
    :param name:
    :param keep_prob:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as sc:
        return tf.nn.dropout(input_op, keep_prob=keep_prob, name='Dropout')


def get_conv_filter(shape, trainable=True, xavier=False):
    """
    convolution layer filter
    :param filter_shape:
    :return:
    """
    if xavier:
        filter = tf.get_variable(shape=shape, initializer=xavier_initializer_conv2d(),
                                dtype=tf.float32, name='Weight',  trainable=trainable)
    else:
        filter = tf.get_variable(shape=shape, name='Weight', trainable=trainable)
    return filter


def get_fc_weight(shape, trainable=True, xavier=False):
    """
    full connect layer weight
    :param weight_shape:
    :return:
    """


    if xavier:
        weight = tf.get_variable(shape=shape, initializer=xavier_initializer(), dtype=tf.float32, name='Weight',
                                 trainable=trainable)
    else:
        weight = tf.get_variable(shape=shape, trainable=trainable, name='Weight')

    return weight


def get_bias(shape, trainable=True):
    """
    get bias
    :param bias_shape:

    :return:
    """
    bias = tf.get_variable(shape=shape, name='Bias', dtype=tf.float32, trainable=trainable)

    return bias





