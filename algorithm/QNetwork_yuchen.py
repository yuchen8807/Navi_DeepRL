'''
Pure Tensorflow implementation. Includes Basic Dueling Double Q network.
yuchen
28.07.2017
'''
from algorithm.policy import *
from algorithm.objectives import *
from algorithm.preprocessors import *
from algorithm.utils import *
from algorithm.core import *
from helper import *

import numpy as np
import sys
import tensorflow as tf

class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, rnn_cell_1, myScope, rnn_cell_2=None):
        #The network recieves a frame(unit8) from the game, flattened into an array (float32).
        #It then resizes it and processes it through four convolutional layers.
        with tf.name_scope(myScope):
            self.imageIn =  tf.placeholder(shape=[None,84,84,num_frames],dtype=tf.float32)
            self.image_permute = tf.transpose(self.imageIn, perm=[0, 3, 1, 2]) # dimension: none*84*84*4 convert to none*4*84*84
            self.image_reshape = tf.reshape(self.image_permute, [-1, 84, 84, 1]) # dimesion may be: 4*84*84
            #self.image_reshape_recoverd = tf.squeeze(tf.gather(tf.reshape(self.image_reshape, [-1, num_frames, 84, 84, 1]), [0]), [0])
            #self.summary_merged = tf.summary.merge([tf.summary.image('image_reshape_recoverd', self.image_reshape_recoverd, max_outputs=num_frames)])
            # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,1])
            self.conv1 = tf.contrib.layers.convolution2d( \
                inputs=self.image_reshape,num_outputs=32,\
                kernel_size=[8,8],stride=[4,4],padding='VALID', \
                activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv1')
            self.conv2 = tf.contrib.layers.convolution2d( \
                inputs=self.conv1,num_outputs=64,\
                kernel_size=[4,4],stride=[2,2],padding='VALID', \
                activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv2')
            self.conv3 = tf.contrib.layers.convolution2d( \
                inputs=self.conv2,num_outputs=64,\
                kernel_size=[3,3],stride=[1,1],padding='VALID', \
                activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv3')
            self.conv4 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv3), h_size, activation_fn=tf.nn.relu)

            # LSTM: the output from the final convolutional layer and send it to a recurrent layer.
            #The input must be reshaped into [batch x trace x units] for rnn processing,
            #and then returned to [batch x units] when sent through the upper levels.
            self.batch_size = tf.placeholder(tf.int32, [])
            self.convFlat = tf.reshape(self.conv4,[self.batch_size, num_frames, h_size]) #dimension: 32*4*512
            self.state_in_1 = rnn_cell_1.zero_state(self.batch_size, tf.float32) # dimension: 32

            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell_1, dtype=tf.float32, \
                initial_state=self.state_in_1, scope=myScope+'_rnn')
            # print "====== self.rnn_outputs ", self.rnn_outputs.get_shape().as_list() # [None, 10, 512]
            self.rnn_output_dim = h_size
            self.rnn_last_output = tf.slice(self.rnn_outputs, [0, num_frames-1, 0], [-1, 1, -1])
            self.rnn = tf.squeeze(self.rnn_last_output, [1])
            # print "========== self.rnn ", self.rnn.get_shape().as_list() #[None, 1024]

            #Double DQN: The output from the recurrent player is then split into separate Value and Advantage streams
            self.ad_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu, scope=myScope+'_fc_advantage_hidden')
            self.Advantage = tf.contrib.layers.fully_connected(self.ad_hidden, num_actions, activation_fn=None, scope=myScope+'_fc_advantage')
            self.value_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu, scope=myScope+'_fc_value_hidden')
            self.Value = tf.contrib.layers.fully_connected(self.value_hidden, 1, activation_fn=None, scope=myScope+'_fc_value')
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))

            self.predict = tf.argmax(self.Qout,1)

            #Loss function: squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32) # onehot vector: 0 in most dimensions, and 1 in a single dimension

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1) # find the Q with high prediction, then sum
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)

            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)
def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)
