# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import numpy as np

LABEL_NUM = 10

def _variable_on_cpu(shape, init, name, dtype=tf.float32, trainable=True):
    with tf.device("/cpu:0"):
        return tf.get_variable(name, shape, dtype=dtype, initializer=init,
                              trainable=trainable)
        
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

def conv2d_layer(x, shape, stride=1, name='conv2d'):
    kernel = weight_variable(shape=shape)
    biases = bias_variable(shape=shape[3])
    conv = tf.nn.conv2d(x, kernel, [1,stride,stride,1], padding='SAME')
    bias = tf.nn.bias_add(conv,biases, name=name)
    return bias

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

def residual_layer(x, width, ker1, red3, ker3, red5, ker5, pool, ker=3, name='residual'):
    inc_out = ker1+ker3+ker5+pool
    
    with tf.variable_scope('inception') as scope:
        inception = inception_layer(x, width, ker1, red3, ker3, red5, ker5, pool, name=scope.name)
        
    with tf.variable_scope('linear') as scope:
        kernel = weight_variable(shape=[ker,ker,inc_out,width])
        biases = bias_variable(shape=width)
        temp = tf.nn.conv2d(inception, kernel, [1,1,1,1], padding='SAME')
        linear = tf.add(temp, biases, name=scope.name)        
    
    return tf.nn.relu(tf.add(linear, x))


def batch_norm(inputs, decay=0.999, center=True, scale=False, epsilon=0.001, activation_fn=None,
               is_training=True, trainable=True):
    with tf.variable_scope('BatchNorm') as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                    inputs.name, params_shape))
    
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center: beta = _variable_on_cpu(shape=params_shape,
                                           init=tf.zeros_initializer, name='beta')
        if scale: gamma = _variable_on_cpu(shape=params_shape,
                                           init=tf.ones_initializer, name='gamma')
        
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean = _variable_on_cpu(shape=params_shape,
                                       init=tf.zeros_initializer, trainable=False, 
                                       name='moving_mean')
        moving_var = _variable_on_cpu(shape=params_shape,
                                      init=tf.ones_initializer, trainable=False,
                                      name='moving_var')

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        if is_training:
            # Calculate the moments based on the individual batch.
            # Use a copy of moving_mean as a shift to compute more reliable moments.
            shift = tf.add(moving_mean, 0)
            mean, variance = tf.nn.moments(inputs, axis, shift=shift)
            with tf.device("/cpu:0"):
                update_moving_mean = moving_mean.assign(tf.div(
                        tf.add(tf.mul(decay,moving_mean),mean),1+decay))
                update_moving_var= moving_var.assign(tf.div(
                        tf.add(tf.mul(decay,moving_var),variance),1+decay))
            with tf.control_dependencies([update_moving_mean,
                                          update_moving_var]):
                mean, variance = tf.identity(mean), tf.identity(variance)
        else:
            mean, variance = moving_mean, moving_var
    
        # Compute batch_normalization.
        outputs = tf.nn.batch_normalization(
                inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs
    
def inference(images, keep_prop, is_training=False, reuse=True):
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(images, [5, 5, 3, 64], name=scope.name)
        norm1 = batch_norm(conv1, is_training=is_training, activation_fn=tf.nn.relu)
        
    with tf.variable_scope('res1') as scope:
        res1 = residual_layer(norm1, 64, 16,24,32,4,8,8, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d_layer(res1, [3, 3, 64, 128], stride=2, name=scope.name)
        norm2 = batch_norm(conv2, is_training=is_training, activation_fn=tf.nn.relu)

    with tf.variable_scope('res2_1') as scope:
        res2_1 = residual_layer(norm2, 128, 32,48,64,8,16,16, name=scope.name)
    with tf.variable_scope('res2') as scope:
        res2 = residual_layer(res2_1, 128, 32,48,64,8,16,16, name=scope.name)

    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d_layer(res2, [3,3,128,256], stride=2, name=scope.name)
        norm3 = batch_norm(conv3, is_training=is_training, activation_fn=tf.nn.relu)
    
    with tf.variable_scope('res3_1') as scope:
        res3_1 = residual_layer(norm3, 256, 64,96,128,16,32,32, name=scope.name)
    with tf.variable_scope('res3') as scope:
        res3 = residual_layer(res3_1, 256, 64,96,128,16,32,32, name=scope.name)

    with tf.variable_scope('avg_pool') as scope:
        avg_pool = tf.nn.avg_pool(res3, ksize=[1,8,8,1], strides=[1,1,1,1],
                                  padding='VALID', name=scope.name)
        norm4 = batch_norm(avg_pool, is_training=is_training)

    with tf.variable_scope('softmax_linear') as scope:
        reshaped = tf.reshape(norm4, [-1, 256])
        weights = weight_variable([256, LABEL_NUM], wd=0.0)
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
