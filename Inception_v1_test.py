#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v1_test.py
# @ Description :
# @ Author alexchung
# @ Time 29/10/2019 PM 13:50


import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from Inception.Inception_v1 import InceptionV1
import numpy as np
from TFRecordProcessing.parse_TFRecord import read_tfrecord
from tensorflow.python.framework import graph_util

original_dataset_dir = '/home/alex/Documents/datasets/dogs_and_cat_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')
record_file = os.path.join(tfrecord_dir, 'image.tfrecords')
model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'inception_v1.pb')


def predict(model_name, image_data, input_op_name, predict_op_name):
    """
    model read and predict
    :param model_name:
    :param image_data:
    :param input_op_name:
    :param predict_op_name:
    :return:
    """
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_name, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        image = image_data.eval()
        input = sess.graph.get_tensor_by_name(name=input_op_name)
        output = sess.graph.get_tensor_by_name(name=predict_op_name)

        predict_softmax = sess.run(fetches=output, feed_dict={input: image})
        predict_label = np.argmax(predict_softmax, axis=1)
        return predict_label





if __name__ == "__main__":
    DATA_SHAPE = [224, 224, 3]
    NUM_CLASS = 2
    BATCH_SIZE = 2
    STEP_NUM = 2
    DECAY_STEPS = 100
    DECAY_RATE = 0.99
    LEARNING_RATE = 1e-2
    KEEP_PROB = 0.8
    GLOBAL_POOL = False
    SPACIAL_SQUEEZE = True
    inception_v1 = InceptionV1(input_shape=DATA_SHAPE, num_classes=NUM_CLASS, batch_size=BATCH_SIZE, decay_rate=DECAY_RATE,
                      decay_steps=DECAY_STEPS, learning_rate=LEARNING_RATE, keep_prob=KEEP_PROB,
                      global_pool=GLOBAL_POOL, spacial_squeeze=SPACIAL_SQUEEZE)
    input_op = inception_v1.raw_input_data.name
    logit_op = inception_v1.logits.name


    # test build network
    def testBuildClissificationNetWork():
        inputs = tf.random_uniform(shape=(BATCH_SIZE, DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2]))
        labels = tf.random_uniform(shape=(BATCH_SIZE, NUM_CLASS))

        with tf.Session() as sess:
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )
            sess.run(init_op)

            inputs, labels = sess.run([inputs, labels])
            feed_dict = inception_v1.fill_feed_dict(image_feed=inputs, label_feed=labels)

            logits = sess.run(fetches=[inception_v1.logits], feed_dict=feed_dict)
            assert logits.shape.shape.tolist() == [BATCH_SIZE, NUM_CLASS]






    # # train and save model
    # sess = tf.Session()
    # with sess.as_default():
    #     images, labels = read_tfrecord(record_file=record_file, batch_size=BATCH_SIZE)
    #     init_op = tf.group(
    #         tf.global_variables_initializer(),
    #         tf.local_variables_initializer()
    #     )
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners()
    #     try:
    #         if not coord.should_stop():
    #             for epoch in range(STEP_NUM):
    #                 image, label = sess.run([images, labels])
    #                 feed_dict = inception_v1.fill_feed_dict(image_feed=image, label_feed=label)
    #
    #                 _, loss_value, train_accuracy = sess.run(fetches=[inception_v1.train,
    #                                                                   inception_v1.loss,
    #                                                                   inception_v1.evaluate_accuracy],
    #                                                          feed_dict=feed_dict)
    #
    #                 print('step {0}:loss value {1}  train accuracy {1}'.format(epoch, loss_value, train_accuracy))
    #
    #             # convert variable to constant
    #             input_graph_def = tf.get_default_graph().as_graph_def()
    #
    #             constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
    #                                                                           [input_op.split(':')[0],
    #                                                                            logit_op.split(':')[0]])
    #             # save to serialize file
    #             with tf.gfile.FastGFile(name=model_name, mode='wb') as f:
    #                 f.write(constant_graph.SerializeToString())
    #
    #     except Exception as e:
    #         print(e)
    #     coord.request_stop()
    #     coord.join(threads)
    # sess.close()
    # print('model training has complete')