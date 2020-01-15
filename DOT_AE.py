# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:37:53 2020

@author: 邹运
"""


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
sys.path.append(r'E:\Zou\simudata') 
import DOT_preprocess,Born
import os
import matplotlib.pyplot as plt

ref_a,pert_a,batch_ys_a,depth_a = DOT_preprocess.dataprocess()
ref_t, meas_t, test_image = DOT_preprocess.test_data()
#pert_a = 1e4*pert_a
#meas_t = 1e4*meas_t
pert_a = (pert_a)/(np.max(pert_a)-np.min(pert_a))
meas_t = (meas_t)/(np.max(meas_t)-np.min(meas_t))
#decoder

learning_rate = 1e-3
learning_rate1 = 1e-5
training_epochs = 250
batch_size = 64
display_step = 1

meas = tf.placeholder(tf.float64, [None,14*9*2], name='meas')
image = tf.placeholder(tf.float64, [None,16,16,3], name='image')


def random_batch(x1_train,x2_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x2_train), batch_size)
    x1_batch = x1_train[rnd_indices]
    x2_batch = x2_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x1_batch,x2_batch, y_batch

initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
weights1 = tf.Variable(initializer([252,64]), name='weights1')
b1 = tf.Variable(tf.constant(0.1, shape=[64],dtype=tf.float64), name="b1")
weights2 = tf.Variable(initializer([64, 128]), name='weights2')
b2 = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float64), name="b2")
weights3 = tf.Variable(initializer([128, 768]), name='weights3')
b3 = tf.Variable(tf.constant(0.1, shape=[768],dtype=tf.float64), name="b3")
## pretrain
#img_r = tf.reshape(meas,(-1,768)) 
l1 = tf.matmul(meas, weights1)+b1
l1 = tf.nn.relu(l1)

l2 = tf.matmul(l1, weights2)+b2
l2 = tf.nn.relu(l2)

img = tf.matmul(l2, weights3)+b3 #img
img_reshape = tf.reshape(image,(-1,768)) 

## encoder
weights11 = tf.Variable(initializer([768,256]), name='weights11')
b11 = tf.Variable(tf.constant(0.1, shape=[256],dtype=tf.float64), name="b11")
weights12 = tf.Variable(initializer([256, 128]), name='weights12')
b12 = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float64), name="b12")
weights13 = tf.Variable(initializer([128, 252]), name='weights13')
b13 = tf.Variable(tf.constant(0.1, shape=[252],dtype=tf.float64), name="b13")


img_r = tf.reshape(image,(-1,768)) 
l3 = tf.matmul(img_r, weights11)+b11
l3 = tf.nn.relu(l3)

l4 = tf.matmul(l3, weights12)+b12
l4 = tf.nn.relu(l4)

m_re = tf.matmul(l4, weights13)+b13
#img_reshape = tf.reshape(img_re,(-1,16,16,3)) 


reg_l1 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-4), tf.trainable_variables())
reg_l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())

cost_meas = 1e0*tf.reduce_mean(tf.abs((m_re - meas)))+ +1e0*reg_l2
cost_img = 1e0*tf.reduce_mean(1e1*tf.pow((img_reshape - img),2))+ +1e0*reg_l2 \
     + 1e-1*tf.reduce_mean(tf.abs(tf.reduce_max((img_reshape),0)-tf.reduce_max((img),0)))


## fine-tune

l1_f = tf.matmul(meas, weights1)+b1
l1_f = tf.nn.relu(l1_f)

l2_f = tf.matmul(l1_f, weights2)+b2
l2_f = tf.nn.relu(l2_f)

img_f = tf.matmul(l2_f, weights3)+b3 #img
img_reshape_f = tf.reshape(img_f,(-1,16,16,3)) 

l3_f = tf.matmul(img_f, weights11)+b11
l3_f = tf.nn.relu(l3_f)

l4_f = tf.matmul(l3_f, weights12)+b12
l4_f = tf.nn.relu(l4_f)

m_re_f = tf.matmul(l4_f, weights13)+b13
cost_meas_f = 1e0*tf.reduce_mean(tf.abs((m_re_f - meas)))+ +1e0*reg_l2

var1 = tf.trainable_variables()[0:5]
var2 = tf.trainable_variables()[6:]

optimizer_i = tf.train.AdamOptimizer(1e-6).minimize(cost_meas_f, var_list=var1)
optimizer_m = tf.train.AdamOptimizer(1e-7).minimize(cost_meas_f, var_list=var2)

optimizer1 = tf.train.AdamOptimizer(1e-4).minimize(cost_img)#, var_list=var1)
optimizer2 = tf.train.AdamOptimizer(5e-5).minimize(cost_meas)#, var_list=var2)
train_op = tf.group(optimizer_i, optimizer_m)
train_op1 = tf.group(optimizer1,optimizer2)
saver = tf.train.Saver()



c=[0]*training_epochs
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(len(pert_a)/batch_size)
    cc = 0
    # Training cycle
    for epoch in range(training_epochs):
        if epoch < 200:
        # Loop over all batches
            for i in range(total_batch):
                ref_train,meas_train, batch_image_train = random_batch(ref_a,pert_a,batch_ys_a, batch_size)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                _,c[epoch] = sess.run([train_op1,cost_img], feed_dict={meas: meas_train,image:batch_image_train})
        else:
            for i in range(total_batch):
                ref_train1,meas_train1, batch_image_train1 = random_batch(ref_a,meas_t,batch_ys_a, 10)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                _,c[epoch] = sess.run([train_op,cost_img], feed_dict={meas: meas_train1,image:batch_image_train1})
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "%.9f" %(c[epoch]), "cost1=", "%.9f" %(cc))
 
    print("Optimization Finished!")
    saver.save(sess,save_path='E:/Zou/simudata/model/fixbackground.ckpt')

    is_train = False

    test_image_show = test_image.reshape((12,256,3))
    out = sess.run([img_f], feed_dict={meas:meas_t,image:test_image})
    out1 = sess.run(img_f, feed_dict={meas: meas_train,image:batch_image_train})
    output = out[0].reshape((12,256,3))# - np.array(minvalue).reshape(len(minvalue),1,1)
    output1 = out1.reshape((batch_size,256,3))
    batch_ys_train_show = batch_image_train.reshape((batch_size,256,3))
plt.plot(c)




