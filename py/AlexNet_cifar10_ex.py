# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import numpy as np

LABEL_NUM = 10

def _variable_on_cpu(name, shape, init):
    with tf.device("/cpu:0"):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
        
def weight_variable(shape, stdev=0.05, wd=0.0):
    init = tf.truncated_normal_initializer(stddev=stdev, dtype=tf.float32)
    var = _variable_on_cpu('wegiths', shape=shape, init=init)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, const=0.1):
    init = tf.constant_initializer(const)
    return _variable_on_cpu('biases', shape=shape, init=init)
    
def inference(images, keep_prop):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable([5, 5, 3, 64])
        biases = bias_variable([64])
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable([5, 5, 64, 64])
        biases = bias_variable([64])
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')
    # norm2
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(norm2, [-1, 8*8*64])
        
        weights = weight_variable([8*8*64, 384], wd=0.004)
        biases = bias_variable([384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        drop3 = tf.nn.dropout(local3, keep_prop)
        
    
    # local4
    with tf.variable_scope('local4') as scope:
        weights = weight_variable([384, 192], wd=0.004)
        biases = bias_variable([192])
        local4 = tf.nn.relu(tf.matmul(drop3, weights) + biases, name=scope.name)
        drop4 = tf.nn.dropout(local4, keep_prop)
        
    
    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable([192, LABEL_NUM], wd=0.0)
        biases = bias_variable([LABEL_NUM])
        softmax_linear = tf.add(tf.matmul(drop4, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def train(total_loss, lr):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op
  
  