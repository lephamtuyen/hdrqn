import numpy as np
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

env = Fourrooms(partial=None,hasObstacle=None,nTasks=2)

class SubController():
    def __init__(self, h_size, rnn_cell, myScope):
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

            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
            self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

            self.rnn, self.rnn_state = tf.nn.dynamic_rnn( \
                inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
            self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            self.AW = tf.Variable(tf.random_normal([h_size // 2, 4]))
            self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

        self.param = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if myScope in v.name]

class sub_buffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            if (len(episode) > trace_length):
                point = np.random.randint(0, len(episode) + 1 - trace_length)
                sampledTraces.append(episode[point:point + trace_length])

        sampledTraces = np.array(sampledTraces)

        return np.reshape(sampledTraces, [-1, 6])

class MetaController():
    def __init__(self, h_size, rnn_cell, num_of_goals, myScope):
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

            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
            self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

            self.rnn, self.rnn_state = tf.nn.dynamic_rnn( \
                inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
            self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            self.AW = tf.Variable(tf.random_normal([h_size // 2, num_of_goals]))
            self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.goals = tf.placeholder(shape=[None], dtype=tf.int32)
        self.goals_onehot = tf.one_hot(self.goals, num_of_goals, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.goals_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        self.loss = tf.reduce_mean(self.td_error)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

        self.param = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if myScope in v.name]

class meta_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            if (len(episode) > trace_length):
                point = np.random.randint(0, len(episode) + 1 - trace_length)
                sampledTraces.append(episode[point:point + trace_length])

        sampledTraces = np.array(sampledTraces)

        return np.reshape(sampledTraces, [-1, 5])

def train():
    #Setting the training parameters
    meta_batch_size = 12
    meta_trace_length = 1
    sub_batch_size = 4 #How many experience traces to use for each training step.
    sub_trace_length = 8 #How long each experience trace will be when training
    update_freq = 5 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    num_episodes = 20000 #How many episodes of game environment to train network with.
    anneling_steps = 10000
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    load_model = False #Whether to load a saved model.
    path = "./hdrqnv1" #The path to save our model to.
    h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    summaryLength = 1000 #Number of epidoes to periodically save for analysis
    tau = 0.001
    num_of_goals = 2

    plot_rewards = np.zeros([3,num_episodes])
    plot_intrinsic = np.zeros([3, num_episodes])
    plot_extrinsic = np.zeros([3, num_episodes])
    plot_steps = np.zeros([3,num_episodes])

    tf.reset_default_graph()
    subCell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    subCellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    metaCell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    metaCellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    subController = SubController(h_size, subCell, 'subController')
    metaController = MetaController(h_size, metaCell, num_of_goals, 'metaController')
    targetSubController = SubController(h_size, subCellT, 'targetSubController')
    targetMetaController = MetaController(h_size, metaCellT, num_of_goals, 'targetMetaController')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    meta_soft_update_op = [tf.assign(target_network_param, (1 - tau) * target_network_param + tau * network_param)
                           for target_network_param, network_param in
                           zip(targetMetaController.param, metaController.param)]
    sub_soft_update_op = [tf.assign(target_network_param, (1 - tau) * target_network_param + tau * network_param)
                          for target_network_param, network_param in
                          zip(targetSubController.param, subController.param)]
    meta_hard_update_op = [tf.assign(target_network_param, network_param)
                           for target_network_param, network_param in
                           zip(targetMetaController.param, metaController.param)]
    sub_hard_update_op = [tf.assign(target_network_param, network_param)
                          for target_network_param, network_param in
                          zip(targetSubController.param, subController.param)]

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
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(init)

            sess.run(meta_hard_update_op)
            sess.run(sub_hard_update_op)

            for i in range(num_episodes):
                meta_e = (endE + max(0., (startE - endE) * (float(num_episodes) - max(0., float(i))) / float(num_episodes)))

                metaEpisodeBuffer = []
                subEpisodeBuffer = []
                total_list = []

                # Reset environment and get first new observation
                state_idx, total, sP = env.reset()
                s = processState(sP, 5808)
                d = False
                rAll = 0
                j = 0
                extrinsic_All = 0
                goal_idx = None

                goal_list = range(num_of_goals)

                meta_interal_state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                while not d and j < max_epLength:

                    # Choose goal
                    if np.random.rand(1) < meta_e or total_steps < pre_train_steps:
                        meta_interal_state = sess.run(metaController.rnn_state, \
                                          feed_dict={metaController.scalarInput: [s], \
                                                     metaController.trainLength: 1, \
                                                     metaController.state_in: meta_interal_state, \
                                                     metaController.batch_size: 1})
                        next_goal_idx = np.random.randint(0, num_of_goals)
                    else:
                        next_goal_idx, meta_interal_state = sess.run([metaController.predict, metaController.rnn_state], \
                                             feed_dict={metaController.scalarInput: [s], \
                                                        metaController.trainLength: 1, \
                                                        metaController.state_in: meta_interal_state, \
                                                        metaController.batch_size: 1})
                        next_goal_idx = next_goal_idx[0]

                    if (next_goal_idx in goal_list):
                        goal_idx = next_goal_idx
                    else:
                        goal_idx = np.random.choice(goal_list)

                    if (goal_idx in goal_list):
                        goal_list.remove(goal_idx)

                    # goal_idx = next_goal_idx if goal_idx is None or goal_idx != next_goal_idx else 1 - goal_idx
                    goal = processState(env.decode_goal(goal_idx), 5808)

                    goal_reached = False
                    extrinsic_reward = 0
                    initial_state = s
                    intrinsic_All = 0

                    sub_interal_state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state
                    while not d and j < max_epLength and goal_reached == False:
                        j += 1
                        # Choose state
                        if np.random.rand(1) < sub_e or total_steps < pre_train_steps:
                            sub_interal_state = sess.run(subController.rnn_state, \
                                              feed_dict={subController.scalarInput: [s], \
                                                         subController.trainLength: 1, \
                                                         subController.goal: [goal], \
                                                         subController.state_in: sub_interal_state, \
                                                         subController.batch_size: 1})
                            a = np.random.randint(0, 4)
                        else:
                            a, sub_interal_state = sess.run([subController.predict, subController.rnn_state], \
                                                 feed_dict={subController.scalarInput: [s], \
                                                            subController.trainLength: 1, \
                                                            subController.goal: [goal], \
                                                            subController.state_in: sub_interal_state, \
                                                            subController.batch_size: 1})
                            a = a[0]

                        next_state_idx, total, s1P, intrinsic_reward, extrinsic_one_step, r, d = env.step(a, goal_idx)
                        s1 = processState(s1P, 5808)
                        total_steps += 1

                        intrinsic_done = 1.0 if goal_idx == next_state_idx else 0.0
                        goal_reached = next_state_idx == goal_idx

                        subEpisodeBuffer.append(np.reshape(np.array([s, a, intrinsic_reward, s1, intrinsic_done, goal]), [1, 6]))
                        total_list.append(total)

                        if total_steps > pre_train_steps:
                            if sub_e > endE:
                                sub_e -= stepDrop

                            if total_steps % (update_freq) == 0:
                                # Reset the recurrent layer's hidden state
                                sub_state_train = (np.zeros([sub_batch_size, h_size]), np.zeros([sub_batch_size, h_size]))

                                trainBatch = subBuffer.sample(sub_batch_size, sub_trace_length)
                                sss = np.vstack(trainBatch[:, 0])
                                sss_next = np.vstack(trainBatch[:, 3])
                                ggg = np.vstack(trainBatch[:, 5])
                                aaaa = trainBatch[:, 1]
                                ddd = trainBatch[:, 4]
                                rrr = trainBatch[:, 2]

                                # Get a random batch of experiences.
                                # Below we perform the Double-DQN update to the target Q-values
                                Q1 = sess.run(subController.predict, feed_dict={ \
                                    subController.scalarInput: sss_next, \
                                    subController.goal: ggg, \
                                    subController.trainLength: sub_trace_length, \
                                    subController.state_in: sub_state_train,
                                    subController.batch_size: sub_batch_size})

                                Q2 = sess.run(targetSubController.Qout, feed_dict={ \
                                    targetSubController.scalarInput: sss_next, \
                                    targetSubController.goal: ggg, \
                                    targetSubController.trainLength: sub_trace_length, \
                                    targetSubController.state_in: sub_state_train,
                                    targetSubController.batch_size: sub_batch_size})

                                end_multiplier = -(ddd - 1)
                                doubleQ = Q2[range(sub_batch_size * sub_trace_length), Q1]
                                targetQ = rrr + (y * doubleQ * end_multiplier)
                                # Update the network with our target values.
                                sess.run(subController.updateModel, \
                                         feed_dict={subController.scalarInput: sss, \
                                                    subController.targetQ: targetQ, \
                                                    subController.goal: ggg, \
                                                    subController.actions: aaaa, \
                                                    subController.trainLength: sub_trace_length, \
                                                    subController.state_in: sub_state_train, \
                                                    subController.batch_size: sub_batch_size})
                                sess.run(sub_soft_update_op)
                                ##############################################################################

                                if (len(metaBuffer.buffer) > meta_batch_size):
                                    # Reset the recurrent layer's hidden state
                                    meta_state_train = (np.zeros([meta_batch_size, h_size]), np.zeros([meta_batch_size, h_size]))

                                    trainBatch = metaBuffer.sample(meta_batch_size, meta_trace_length)
                                    sss = np.vstack(trainBatch[:, 0])
                                    sss_next = np.vstack(trainBatch[:, 3])
                                    ggg = trainBatch[:, 1]
                                    ddd = trainBatch[:, 4]
                                    rrr = trainBatch[:, 2]

                                    # Get a random batch of experiences.
                                    # Below we perform the Double-DQN update to the target Q-values
                                    Q1 = sess.run(metaController.predict, feed_dict={ \
                                        metaController.scalarInput: sss_next, \
                                        metaController.trainLength: meta_trace_length, \
                                        metaController.state_in: meta_state_train,
                                        metaController.batch_size: meta_batch_size})

                                    Q2 = sess.run(targetMetaController.Qout, feed_dict={ \
                                        targetMetaController.scalarInput: sss_next, \
                                        targetMetaController.trainLength: meta_trace_length, \
                                        targetMetaController.state_in: meta_state_train,
                                        targetMetaController.batch_size: meta_batch_size})

                                    end_multiplier = -(ddd - 1)
                                    doubleQ = Q2[range(meta_batch_size * meta_trace_length), Q1]
                                    targetQ = rrr + (y * doubleQ * end_multiplier)
                                    # Update the network with our target values.
                                    sess.run(metaController.updateModel, \
                                             feed_dict={metaController.scalarInput: sss, \
                                                        metaController.targetQ: targetQ, \
                                                        metaController.goals: ggg, \
                                                        metaController.trainLength: meta_trace_length, \
                                                        metaController.state_in: meta_state_train, \
                                                        metaController.batch_size: meta_batch_size})

                                    sess.run(meta_soft_update_op)

                        rAll += r
                        extrinsic_reward += extrinsic_one_step
                        intrinsic_All += intrinsic_reward
                        s = s1

                    metaEpisodeBuffer.append(np.reshape(np.array([initial_state, goal_idx, extrinsic_reward, s1, d]), [1, 5]))

                    extrinsic_All += extrinsic_reward
                    intrinsicList.append(intrinsic_All)

                    if (len(subEpisodeBuffer) > sub_trace_length):
                        subBuffer.add(subEpisodeBuffer)

                if (len(metaEpisodeBuffer) > meta_trace_length):
                    metaBuffer.add(metaEpisodeBuffer)

                jList.append(j)
                rList.append(rAll)

                plot_rewards[runs, i] = rAll
                plot_steps[runs, i] = j
                plot_intrinsic[runs, i] = intrinsic_All
                plot_extrinsic[runs, i] = extrinsic_All

                if len(rList) % summaryLength == 0 and len(rList) != 0:

                    print (i, np.mean(rList[-summaryLength:]))
                    print (np.mean(intrinsicList[-summaryLength:]))

                    np.save(path + "/reward_partial_" + str(
                        env.partial) + "_meta_" + str(
                        meta_trace_length) + "_sub_" + str(
                        sub_trace_length) + "_obstacle_" + str(
                        env.hasObstacle) + "_task_" + str(env.nTasks),
                            plot_rewards)
                    np.save(path + "/intrinsic_partial_" + str(env.partial) + "_meta_" + str(
                        meta_trace_length) + "_sub_" + str(sub_trace_length) + "_obstacle_" + str(
                        env.hasObstacle) + "_task_" + str(env.nTasks), plot_intrinsic)
                    np.save(path + "/extrinsic_partial_" + str(env.partial) + "_meta_" + str(
                        meta_trace_length) + "_sub_" + str(sub_trace_length) + "_obstacle_" + str(
                        env.hasObstacle) + "_task_" + str(env.nTasks), plot_extrinsic)
                    np.save(
                        path + "/steps_partial_" + str(env.partial) + "_meta_" + str(meta_trace_length) + "_sub_" + str(
                            sub_trace_length) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_steps)

if __name__ == '__main__':
    train()
