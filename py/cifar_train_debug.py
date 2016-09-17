# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:43:00 2016

@author: bong
"""

from cifar_input import cifar10_input
import ResNet_cifar10 as model
import tensorflow as tf

BATCH_SIZE=50
EVAL_SIZE=1000

def tower_loss(scope):
    train_data, _= cifar10_input()
    images, labels = train_data.dist_batch(BATCH_SIZE,20000)
    
    logits = model.inference(images, keep_prob)
    _ = model.loss(logits, labels)
    
    losses = tf.get_collection('losses', scope)
    
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss
    

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_data, test_data = cifar10_input()
        
        images, labels = test_data.batch(EVAL_SIZE)

        keep_prob = tf.placeholder(tf.float32)
        lr = tf.placeholder(tf.float32)

        logit = model.inference(images, keep_prob)
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logit, labels, 1),tf.float32))
        
        opt = tf.GradientDescentOptimizer(lr)

        with tf.device('/gpu:0'):
            logit = model.inference(x, keep_prob)
            loss = model.loss(logit, y_)
            grads = opt.compute_gradients(loss)    
    
        train_op = opt.apply_gradients(grads)
        sess = tf.Session()
        
        rate = 0.1
        mean_loss = 0.
        least = 10.
        count = 0.
    
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)
        for i in range(20000):
            _, cross_entropy = sess.run([train_op, loss], feed_dict = {keep_prob: 0.5,
                                                                       lr:rate})
            mean_loss = mean_loss + cross_entropy/50
    
            if i%50 == 0 and i>0:
                print("step %d, cross_entropy %g"%(i, mean_loss))
      
                if mean_loss > least:
                    count = count+1
                    if count > 4:
                        rate = rate/2.0
                        count = 0
                    else:
                        count = 0
                        least = mean_loss
                mean_loss=0
        
            if i%250 == 0 and i>0:
                eval_accuracy = sess.run(accuracy)
                print("test accuracy %g"%(eval_accuracy))
