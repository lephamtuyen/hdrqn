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
# np.random.seed(123)
# tf.set_random_seed(123)

env = Fourrooms(partial=5,hasObstacle=None,nTasks=2)

class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope):
        self.scalarInput = tf.placeholder(shape=[None, 5808], dtype=tf.float32)

        self.scalarInputReshape = tf.reshape(self.scalarInput, shape=[-1, 44, 44, 3])

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


class experience_buffer():
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
    batch_size = 4 #How many experience traces to use for each training step.
    trace_length = 8 #How long each experience trace will be when training
    update_freq = 5 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    num_episodes = 20000 #How many episodes of game environment to train network with.
    anneling_steps = 10000
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    path = "./drqn" #The path to save our model to.
    h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    max_epLength = 50 #The max allowed length of our episode.
    summaryLength = 500 #Number of epidoes to periodically save for analysis
    tau = 0.001

    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main')
    targetQN = Qnetwork(h_size, cellT, 'target')

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

            updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.

            for i in range(num_episodes):
                episodeBuffer = []
                total_list = []
                # Reset environment and get first new observation
                state_idx, _, sP = env.reset()
                s = processState(sP, 5808)
                d = False
                rAll = 0
                j = 0
                internal_state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state
                while not d and j < max_epLength:
                    j += 1
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        internal_state = sess.run(mainQN.rnn_state, \
                                          feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1,
                                                     mainQN.state_in: internal_state, mainQN.batch_size: 1})
                        a = np.random.randint(0, 4)
                    else:
                        a, internal_state = sess.run([mainQN.predict, mainQN.rnn_state], \
                                             feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1,
                                                        mainQN.state_in: internal_state, mainQN.batch_size: 1})
                        a = a[0]

                    next_state_idx, total, s1P, intrinsic_reward, extrinsic_one_step, r, d = env.step(a, 0)
                    s1 = processState(s1P, 5808)
                    total_steps += 1
                    episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop

                        if total_steps % (update_freq) == 0:
                            updateTarget(targetOps, sess)
                            # Reset the recurrent layer's hidden state
                            state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                            trainBatch = myBuffer.sample(batch_size, trace_length)
                            sss = np.vstack(trainBatch[:, 0])
                            sss_next = np.vstack(trainBatch[:, 3])
                            aaa = trainBatch[:, 1]
                            ddd = trainBatch[:, 4]
                            rrr = trainBatch[:, 2]

                            # Get a random batch of experiences.
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={ \
                                mainQN.scalarInput: sss_next, \
                                mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size})

                            Q2 = sess.run(targetQN.Qout, feed_dict={ \
                                targetQN.scalarInput: sss_next, \
                                targetQN.trainLength: trace_length, targetQN.state_in: state_train,
                                targetQN.batch_size: batch_size})

                            end_multiplier = -(ddd - 1)
                            doubleQ = Q2[range(batch_size * trace_length), Q1]
                            targetQ = rrr + (y * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            sess.run(mainQN.updateModel, \
                                     feed_dict={mainQN.scalarInput: sss,
                                                mainQN.targetQ: targetQ,
                                                mainQN.actions: aaa, mainQN.trainLength: trace_length,
                                                mainQN.state_in: state_train, mainQN.batch_size: batch_size})
                    rAll += r
                    s = s1

                if (len(episodeBuffer) > trace_length):
                    myBuffer.add(episodeBuffer)
                jList.append(j)
                rList.append(rAll)

                plot_rewards[runs, i] = rAll
                plot_steps[runs, i] = j

                if len(rList) % summaryLength == 0 and len(rList) != 0:
                    print (i, np.mean(rList[-summaryLength:]), e)

            np.save(path + "/reward_partial_" + str(env.partial) + "_lstm_" + str(
                trace_length) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_rewards)
            np.save(path + "/steps_partial_" + str(env.partial) + "_lstm_" + str(
                trace_length) + "_obstacle_" + str(env.hasObstacle) + "_task_" + str(env.nTasks), plot_steps)

if __name__ == '__main__':
    train()
