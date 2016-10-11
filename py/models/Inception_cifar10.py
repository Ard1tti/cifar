# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import numpy as np

LABEL_NUM = 10

def _variable_on_cpu(shape, init, name):
    with tf.device("/cpu:0"):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
        
def weight_variable(shape, stdev=0.05, wd=0.0):
    init = tf.truncated_normal_initializer(stddev=stdev, dtype=tf.float32)
    var = _variable_on_cpu(shape=shape, init=init, name='weights')
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, const=0.1):
    init = tf.constant_initializer(const)
    return _variable_on_cpu(shape=shape, init=init, name='biases')

def conv2d_layer(x, shape, name='conv2d'):
    kernel = weight_variable(shape=shape)
    biases = bias_variable(shape=shape[3])
    conv = tf.nn.conv2d(x, kernel, [1,1,1,1], padding='SAME')
    bias = tf.nn.bias_add(conv,biases)
    return tf.nn.relu(bias, name=name)

def max_pool_layer(x, stride=2, name='max_pool'):
    return tf.nn.max_pool(x, name=name, ksize=[1,3,3,1],
                          strides=[1,stride,stride,1],padding='SAME')

def inception_layer(x, width, ker1, red3, ker3, red5, ker5, pool, name='inception'):
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(x, [1, 1, width, ker1], name=scope.name)
    
    with tf.variable_scope('red3') as scope:
        red_3 = conv2d_layer(x, [1, 1, width, red3], name=scope.name)
    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d_layer(red_3, [3, 3, red3, ker3], name=scope.name)
        
    with tf.variable_scope('red5') as scope:
        red_5 = conv2d_layer(x, [1, 1, width, red5], name=scope.name)
    with tf.variable_scope('conv5') as scope:
        conv5 = conv2d_layer(red_5, [5, 5, red5, ker5], name=scope.name)
    
    with tf.variable_scope('pooled') as scope:
        pooled = max_pool_layer(x, stride=1, name=scope.name)
    with tf.variable_scope('pool_red') as scope:
        pool_conv = conv2d_layer(pooled, [1, 1, width, pool], name=scope.name)
    
    return tf.concat(3, [conv1, conv3, conv5, pool_conv], name=name)
    
    

    
def inference(images, keep_prop):
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(images, [5, 5, 3, 64], name=scope.name)
    
    with tf.variable_scope('pool1') as scope:
        pool1 = max_pool_layer(conv1, name=scope.name)

    with tf.variable_scope('incep1') as scope:
        incep1 = inception_layer(pool1, 64, 32,48,64,8,16,16, name=scope.name)

    with tf.variable_scope('incep2') as scope:
        incep2 = inception_layer(incep1, 128, 64,64,96,16,48,32, name=scope.name)

    with tf.variable_scope('pool2') as scope:
        pool2 = max_pool_layer(incep2, name=scope.name)

    with tf.variable_scope('avg_pool') as scope:
        avg_pool = tf.nn.avg_pool(pool2, ksize=[1,8,8,1], strides=[1,1,1,1],
                                  padding='VALID', name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        reshaped = tf.reshape(avg_pool, [-1, 240])
        weights = weight_variable([240, LABEL_NUM], wd=0.0)
        biases = bias_variable([LABEL_NUM])
        softmax_linear = tf.add(tf.matmul(reshaped, weights), biases, name=scope.name)

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