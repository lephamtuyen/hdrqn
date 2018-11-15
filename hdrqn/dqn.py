import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

from utils.utils import *
from envs.fourrooms import Fourrooms

env = Fourrooms(partial=5,hasObstacle=None,nTasks=2)

class Qnetwork():
    def __init__(self, h_size, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 5808], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 44, 44, 3])

        self.conv1 = slim.convolution2d( \
            inputs=self.imageIn, num_outputs=32, \
            kernel_size=[3, 3], stride=[4, 4], padding='VALID', \
            biases_initializer=None, scope=myScope + '_critic_conv1')
        self.conv2 = slim.convolution2d( \
            inputs=self.conv1, num_outputs=32, \
            kernel_size=[3, 3], stride=[2, 2], padding='VALID', \
            biases_initializer=None, scope=myScope + '_critic_conv2')
        self.conv3 = slim.convolution2d( \
            inputs=self.conv2, num_outputs=32, \
            kernel_size=[3, 3], stride=[1, 1], padding='VALID', \
            biases_initializer=None, scope=myScope + '_critic_conv3')
        self.conv4 = slim.convolution2d( \
            inputs=self.conv3, num_outputs=h_size, \
            kernel_size=[3, 3], stride=[1, 1], padding='VALID', \
            biases_initializer=None, scope=myScope + '_critic_conv4')

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.action_predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name=myScope + "_actions")
        self.actions_oneshot = tf.one_hot(self.actions, 4, dtype=tf.float32, name=myScope + "_actions_oneshot")
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_oneshot), axis=1, name=myScope + "_actions_q")

        self.loss_critic = tf.reduce_mean(tf.square(self.targetQ - self.Q), name=myScope + "_loss")

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_critic = self.trainer.minimize(self.loss_critic, name=myScope + "_update_critic")

class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        reshaped_samples = np.reshape(samples, [batch_size, 5])
        return reshaped_samples

def train():
    #Setting the training parameters
    batch_size = 32 #How many experience traces to use for each training step.
    update_freq = 5 #How often to perform a training step.
    gamma = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    anneling_steps = 10000 #How many steps of training to reduce startE to endE.
    num_episodes = 20000 #How many episodes of game environment to train network with.
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    path = "./dqn" #The path to save our model to.
    h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    summaryLength = 100 #Number of epidoes to periodically save for analysis
    tau = 0.001

    tf.reset_default_graph()
    # We define the cells for the primary and target q-networks
    mainQN = Qnetwork(h_size, 'main')
    targetQN = Qnetwork(h_size, 'target')

    init = tf.global_variables_initializer()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    plot_rewards = np.zeros([3, num_episodes])
    plot_steps = np.zeros([3, num_episodes])
    for runs in range(3):
        myBuffer = experience_buffer()

        # Set the rate of random action decrease.
        e = startE
        stepDrop = (startE - endE) / anneling_steps

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0

        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        with tf.Session() as sess:

            sess.run(init)

            # Set the target network to be equal to the primary network.
            updateTarget(targetOps, sess)

            for i in range(num_episodes):
                state_idx, _, s = env.reset()
                s = processState(s, 5808)
                d = False
                rAll = 0

                # step counter
                j = 0

                while not d and j < max_epLength:
                    j += 1

                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        a = np.random.randint(0, 4)
                    else:
                        a = sess.run(mainQN.action_predict,
                                             feed_dict={
                                                 mainQN.scalarInput: [s]
                                             })
                        a = a[0]

                    next_state_idx, full_s, s1P, intrinsic_reward, extrinsic_one_step, r, d = env.step(a, 0)
                    s1 = processState(s1P, 5808)
                    total_steps += 1
                    myBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop

                        if total_steps % (update_freq) == 0:
                            # Get a random batch of experiences.
                            trainBatch = myBuffer.sample(batch_size)
                            # Below we perform the Double-DQN update to the target Q-values
                            end_multiplier = -(trainBatch[:, 4] - 1)

                            Q1 = sess.run(mainQN.action_predict, feed_dict={
                                mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={
                                targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                            doubleQ = Q2[range(batch_size), Q1]

                            targetQ = trainBatch[:, 2] + (gamma * end_multiplier * doubleQ)

                            # Update the network with our target values.
                            sess.run(mainQN.update_critic,
                                         feed_dict={
                                             mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                             mainQN.targetQ: targetQ,
                                             mainQN.actions: trainBatch[:, 1]})

                            updateTarget(targetOps, sess)

                    rAll += r
                    s = s1

                jList.append(j)
                rList.append(rAll)

                plot_rewards[runs, i] = rAll
                plot_steps[runs, i] = j

                if len(rList) % summaryLength == 0 and len(rList) != 0:
                    print (i, np.mean(rList[-summaryLength:]), e)

                np.save(path + "/reward_partial_" + str(env.partial) + "_batch_" + str(
                    batch_size) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_rewards)
                np.save(path + "/steps_partial_" + str(env.partial) + "_batch_" + str(
                    batch_size) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_steps)

if __name__ == '__main__':
    train()
