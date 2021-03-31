# -*- coding: utf-8 -*-
'''This is the model code of CNN_SV. Because part of the input layer processing is not convenient to use tensorflow, it is placed in read.py'''
import tensorflow as tf
import numpy as np
import pickle

class Flags:#Store all the parameters required by the model
    def __init__(self):
        self.train_dir = '/home/ubuntu/NLP_Project/ForSICK/dataset/SICK_train.txt' #used for train
        self.test_dir = '/home/ubuntu/NLP_Project/ForSICK/dataset/SICK_test.txt'  #used for test
        self.embeddings_dir ='/home/ubuntu/NLP_Project/ForSICK/dataset/sample.txt' #used for user input
        self.len_of_word_vector = 100
        self.len_of_sentence_vector = 100
        self.filter_para = 5
        self.batch_size = 50
        self.max_k = 5


def conv2d(x, shape): #x is input
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
    conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    return conv

def cnn_sv(x,FLAGS):
    conv1 = conv2d(x, [FLAGS.filter_para, FLAGS.filter_para, FLAGS.len_of_word_vector, FLAGS.len_of_sentence_vector])

    pool2_flat = tf.reshape(conv1, [FLAGS.batch_size,-1, FLAGS.len_of_sentence_vector])#Turn a two-dim graph into a one-dime vector group, three-dim
    pool2_flat = tf.transpose(pool2_flat,[0,2,1])#Deformation, convenient for max-k pooling
    values, indices = tf.nn.top_k(pool2_flat, k=FLAGS.max_k)

    norm = tf.sqrt(tf.reduce_sum(tf.square(values),1,keep_dims=True))
    sv = values / norm #sv contain 3 dim, and one sv have 3 vector
    return sv
