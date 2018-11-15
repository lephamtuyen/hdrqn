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

class SubController():
    def __init__(self, h_size, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 5808], dtype=tf.float32)
        self.goal = tf.placeholder(shape=[None, 5808], dtype=tf.float32)

        self.scalarInputReshape = tf.reshape(self.scalarInput, shape=[-1, 44, 44, 3])
        self.goalReshape = tf.reshape(self.goal, shape=[-1, 44, 44, 3])

        self.input = tf.concat([self.goalReshape, self.scalarInputReshape], 3)

        with tf.variable_scope(myScope):
            self.conv1 = slim.convolution2d( \
                inputs=self.input, num_outputs=32, \
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

        self.param = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if myScope in v.name]

class sub_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        reshaped_samples = np.reshape(samples, [batch_size, 6])
        return reshaped_samples

class MetaController():
    def __init__(self, h_size, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 5808], dtype=tf.float32)

        self.scalarInputReshape = tf.reshape(self.scalarInput, shape=[-1, 44, 44, 3])

        with tf.variable_scope(myScope):
            self.conv1 = slim.convolution2d( \
                inputs=self.scalarInputReshape, num_outputs=32, \
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
            self.AW = tf.Variable(xavier_init([h_size // 2, 2]))
            self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.goal_predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.goals = tf.placeholder(shape=[None], dtype=tf.int32, name=myScope + "_goals")
        self.goals_oneshot = tf.one_hot(self.goals, env.nTasks, dtype=tf.float32, name=myScope + "_goals_oneshot")
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.goals_oneshot), axis=1, name=myScope + "_goals_q")

        self.loss_critic = tf.reduce_mean(tf.square(self.targetQ - self.Q), name=myScope + "_loss")

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_critic = self.trainer.minimize(self.loss_critic, name=myScope + "_update_critic")

        self.param = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if myScope in v.name]

class meta_buffer():
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
    sub_batch_size = 32 #How many experience traces to use for each training step.
    meta_batch_size = 32  # How many experience traces to use for each training step.
    update_freq = 5 #How often to perform a training step.
    gamma = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    anneling_steps = 10000 #How many steps of training to reduce startE to endE.
    num_episodes = 50000 #How many episodes of game environment to train network with.
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    path = "./hdqn" #The path to save our model to.
    h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    summaryLength = 100 #Number of epidoes to periodically save for analysis
    tau = 0.001

    tf.reset_default_graph()
    # We define the cells for the primary and target q-networks
    subController = SubController(h_size, 'subController')
    targetSubController = SubController(h_size, 'targetSubController')
    metaController = MetaController(h_size, 'metaController')
    targetMetaController = MetaController(h_size, 'targetMetaController')

    init = tf.global_variables_initializer()

    meta_soft_update_op = [tf.assign(target_network_param, (1 - tau) * target_network_param + tau * network_param)
                      for target_network_param, network_param in zip(targetMetaController.param, metaController.param)]
    sub_soft_update_op = [tf.assign(target_network_param, (1 - tau) * target_network_param + tau * network_param)
                     for target_network_param, network_param in zip(targetSubController.param, subController.param)]
    meta_hard_update_op = [tf.assign(target_network_param, network_param)
                           for target_network_param, network_param in
                           zip(targetMetaController.param, metaController.param)]
    sub_hard_update_op = [tf.assign(target_network_param, network_param)
                          for target_network_param, network_param in
                          zip(targetSubController.param, subController.param)]

    plot_rewards = np.zeros([3, num_episodes])
    plot_intrinsic = np.zeros([3, num_episodes])
    plot_extrinsic = np.zeros([3, num_episodes])
    plot_steps = np.zeros([3, num_episodes])

    for runs in range(3):
        subBuffer = sub_buffer()
        metaBuffer = meta_buffer()

        # Set the rate of random action decrease.
        sub_e = startE
        stepDrop = (startE - endE) / anneling_steps

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0
        intrinsicList = []

        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        with tf.Session() as sess:

            sess.run(init)

            # Set the target network to be equal to the primary network.
            sess.run(meta_hard_update_op)
            sess.run(sub_hard_update_op)

            for i in range(num_episodes):
                meta_e = (endE + max(0., (startE - endE) * (float(num_episodes) - max(0., float(i))) / float(num_episodes)))

                total_list = []
                state_idx, _, s = env.reset()
                s = processState(s, 5808)
                d = False
                rAll = 0
                j = 0
                extrinsic_All = 0

                goal_idx = None
                goal_list = range(env.nTasks)

                while not d and j < max_epLength:
                    # Choose goal
                    if np.random.rand(1) < meta_e or total_steps < pre_train_steps:
                        next_goal_idx = np.random.randint(0, 2)
                    else:
                        next_goal_idx = sess.run(metaController.goal_predict,
                                     feed_dict={
                                         metaController.scalarInput: [s]
                                     })
                        next_goal_idx = next_goal_idx[0]

                    if (next_goal_idx in goal_list):
                        goal_idx = next_goal_idx
                    else:
                        goal_idx = np.random.choice(goal_list)

                    if (goal_idx in goal_list):
                        goal_list.remove(goal_idx)

                    goal = processState(env.decode_goal(goal_idx), 5808)

                    goal_reached = False
                    extrinsic_reward = 0
                    initial_state = s
                    intrinsic_All = 0

                    while not d and j < max_epLength and goal_reached == False:
                        j += 1

                        # Choose an action by greedily (with e chance of random action) from the Q-network
                        if np.random.rand(1) < meta_e or total_steps < pre_train_steps:
                            a = np.random.randint(0, 4)
                        else:
                            a = sess.run(subController.action_predict,
                                                 feed_dict={
                                                     subController.scalarInput: [s],
                                                     subController.goal: [goal]
                                                 })
                            a = a[0]

                        next_state_idx, total, s1P, intrinsic_reward, extrinsic_one_step, r, d = env.step(a, goal_idx)
                        s1 = processState(s1P, 5808)
                        total_steps += 1

                        intrinsic_done = 1.0 if goal_idx == next_state_idx else 0.0
                        goal_reached = next_state_idx == goal_idx

                        subBuffer.add(np.reshape(np.array([s, a, intrinsic_reward, s1, intrinsic_done, goal]), [1, 6]))
                        total_list.append(total)

                        if total_steps > pre_train_steps:
                            if sub_e > endE:
                                sub_e -= stepDrop

                            if total_steps % (update_freq) == 0:
                                trainBatch = subBuffer.sample(sub_batch_size)
                                rrr = trainBatch[:, 2]
                                sss = np.vstack(trainBatch[:, 0])
                                aaa = trainBatch[:, 1]
                                sss_next = np.vstack(trainBatch[:, 3])
                                ggg = np.vstack(trainBatch[:, 5])

                                # Below we perform the Double-DQN update to the target Q-values
                                end_multiplier = -(trainBatch[:, 4] - 1)

                                Q1 = sess.run(subController.action_predict, feed_dict={
                                    subController.scalarInput: sss_next,
                                    subController.goal: ggg})
                                Q2 = sess.run(targetSubController.Qout, feed_dict={
                                    targetSubController.scalarInput: sss_next,
                                    targetSubController.goal: ggg})

                                doubleQ = Q2[range(sub_batch_size), Q1]

                                targetQ = rrr + (gamma * end_multiplier * doubleQ)

                                # Update the network with our target values.
                                sess.run(subController.update_critic,
                                             feed_dict={
                                                 subController.scalarInput: sss,
                                                 subController.goal: ggg,
                                                 subController.targetQ: targetQ,
                                                 subController.actions: aaa})

                                sess.run(sub_soft_update_op)
                                ##############################################################################
                                trainBatch = metaBuffer.sample(meta_batch_size)  # Get a random batch of experiences.
                                rrr = trainBatch[:, 2]
                                sss = np.vstack(trainBatch[:, 0])
                                ggg = trainBatch[:, 1]
                                sss_next = np.vstack(trainBatch[:, 3])

                                # Below we perform the Double-DQN update to the target Q-values
                                end_multiplier = -(trainBatch[:, 4] - 1)

                                Q1 = sess.run(metaController.goal_predict, feed_dict={
                                    metaController.scalarInput: sss_next})
                                Q2 = sess.run(targetMetaController.Qout, feed_dict={
                                    targetMetaController.scalarInput: sss_next})

                                doubleQ = Q2[range(meta_batch_size), Q1]

                                targetQ = rrr + (0.99 * end_multiplier * doubleQ)

                                # Update the network with our target values.
                                sess.run(metaController.update_critic,
                                         feed_dict={
                                             metaController.scalarInput: sss,
                                             metaController.targetQ: targetQ,
                                             metaController.goals: ggg})

                                sess.run(meta_soft_update_op)

                        extrinsic_reward += extrinsic_one_step
                        rAll += r
                        intrinsic_All += intrinsic_reward
                        extrinsic_All += extrinsic_reward
                        s = s1

                    metaBuffer.add(np.reshape(np.array([initial_state, goal_idx, extrinsic_reward, s1, d]), [1, 5]))

                    intrinsicList.append(intrinsic_All)

                jList.append(j)
                rList.append(rAll)

                plot_rewards[runs, i] = rAll
                plot_steps[runs, i] = j
                plot_intrinsic[runs, i] = intrinsic_All
                plot_extrinsic[runs, i] = extrinsic_All

                if len(rList) % summaryLength == 0 and len(rList) != 0:
                    print (i, np.mean(rList[-summaryLength:]))
                    print (np.mean(intrinsicList[-summaryLength:]))

            np.save(path + "/reward_partial_" + str(env.partial) + "_metabatch_" + str(meta_batch_size) + "_subbatch_" + str(
                sub_batch_size) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_rewards)
            np.save(path + "/intrinsic_partial_" + str(env.partial) + "_metabatch_" + str(meta_batch_size) + "_subbatch_" + str(
                sub_batch_size) + "_obstacle_" + str(
                env.hasObstacle) + "_task_" + str(env.nTasks), plot_intrinsic)
            np.save(path + "/extrinsic_partial_" + str(env.partial) + "_metabatch_" + str(meta_batch_size) + "_subbatch_" + str(
                sub_batch_size) + "_obstacle_" + str(
                env.hasObstacle) + "_task_" + str(env.nTasks), plot_extrinsic)
            np.save(path + "/steps_partial_" + str(env.partial) + "_metabatch_" + str(meta_batch_size) + "_subbatch_" + str(
                sub_batch_size) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_steps)

if __name__ == '__main__':
    train()
