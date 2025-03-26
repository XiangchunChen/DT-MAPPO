"""
Proximal Policy Optimization (PPO) implementation modified from DDPG code.
PPO is a policy gradient method for reinforcement learning.

Using:
tensorflow 1.14.0
gym 0.15.3
"""
import math
import random

import numpy as np
import tensorflow as tf
import pandas as pd
import logging

np.random.seed(1)

#####################  hyper parameters  ####################
MAX_EPISODES = 1000

# PPO specific parameters
LR_A = 0.0003  # learning rate for actor
LR_C = 0.0003  # learning rate for critic
GAMMA = 0.99   # discount factor
EPSILON = 0.2  # clipping parameter
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
OUTPUT_GRAPH = False
UPDATE_STEPS = 10  # update steps for PPO
EPOCHS = 10  # training epochs for PPO


###############################  PPO  ####################################
class PPO(object):  # Keep class name for compatibility
    def __init__(self, a_dim, s_dim, a_bound, scope):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.ADV = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.OLD_PI = tf.placeholder(tf.float32, [None, a_dim * 2], 'old_pi')  # mean and std

        with tf.variable_scope(scope):
            with tf.variable_scope('Actor'):
                # PPO uses a stochastic policy
                self.mu, self.sigma = self._build_a(self.S, scope='eval', trainable=True)
                self.old_mu, self.old_sigma = tf.split(self.OLD_PI, num_or_size_splits=2, axis=1)

                # Continuous action distribution
                self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
                self.old_normal_dist = tf.distributions.Normal(self.old_mu, self.old_sigma)

                # Action sampling for exploration
                self.a = self.normal_dist.sample()
                self.a = tf.clip_by_value(self.a, -self.a_bound, self.a_bound)

            with tf.variable_scope('Critic'):
                self.v = self._build_c(self.S, scope='eval', trainable=True)
                self.v_ = self._build_c(self.S_, scope='target', trainable=False)

        # Get parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/target')

        # Define soft update operation (only for value network in PPO)
        self.soft_replace = [tf.assign(t, t) for t in self.ct_params]  # Placeholder for compatibility

        # PPO loss and optimization
        with tf.variable_scope('ppo_loss'):
            # Value loss (MSE)
            self.v_target = self.R + GAMMA * self.v_
            self.value_loss = tf.reduce_mean(tf.square(self.v_target - self.v))
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.value_loss, var_list=self.ce_params)

            # Policy loss (PPO clipped objective)
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(self.normal_dist.log_prob(self.A) - self.old_normal_dist.log_prob(self.A))
                ratio = tf.reduce_mean(ratio, axis=1, keepdims=True)  # 确保维度正确
                surr = ratio * self.ADV
                clipped_surr = tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * self.ADV
                self.policy_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))
                self.entropy = tf.reduce_mean(self.normal_dist.entropy())  # Entropy bonus for exploration
                self.actor_loss = self.policy_loss - 0.01 * self.entropy  # Entropy bonus
                self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.actor_loss, var_list=self.ae_params)

        self.a_cost = []
        self.c_cost = []
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_a(self, s, scope, trainable):
        # Creating a stochastic policy network that outputs mean and standard deviation
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)

            # Mean of the Gaussian policy
            mu = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='mu', trainable=trainable)
            mu = tf.multiply(mu, self.a_bound, name='scaled_mu')

            # Standard deviation (log std to ensure positivity)
            log_sigma = tf.layers.dense(net, self.a_dim, activation=None, name='log_sigma', trainable=trainable)
            sigma = tf.exp(log_sigma) + 0.1  # Add small constant for stability

            return mu, sigma

    def _build_c(self, s, scope, trainable):
        # Value function network
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l5', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)

    def plot_cost(self):
        f1 = open("result/a_cost.csv", "w")
        for i in range(len(self.a_cost)):
            f1.write(str(i)+","+str(self.a_cost[i])+"\n")
        f1.close()

        f1 = open("result/c_cost.csv", "w")
        for i in range(len(self.c_cost)):
            f1.write(str(i)+","+str(self.c_cost[i])+"\n")
        f1.close()

    def choose_action(self, s):
        # 保持与原函数相同的输入输出接口
        # 但内部实现改为基于PPO的策略采样
        mu, sigma = self.sess.run([self.mu, self.sigma], {self.S: s[np.newaxis, :]})
        # 从高斯分布中采样动作
        action = np.random.normal(mu[0], sigma[0])
        # TODO: 将动作剪裁到合法范围内,合法范围为[1, 8]
        action = np.clip(action, 1, self.a_bound)

        # 保持与原函数相同的返回格式
        if np.isnan(action[0]):
            print("Warning: action[0] is NaN")
            action[0] = 1  # 或者其他默认值
        device_action = math.ceil(action[0])
        # if device_action > 8 or device_action < 1:
        #     device_action = random.randint(1, 8)
        if np.isnan(action[1]):
            print("Warning: action[1] is NaN")
            action[1] = 1  # 或者其他默认值
        bandwidth_action = math.ceil(action[1])
        # if bandwidth_action > 10 or bandwidth_action <= 0:
        #     bandwidth_action = random.randint(1, 10)
        if np.isnan(action[2]):
            print("Warning: action[2] is NaN")
            action[2] = 1
        waitTime = math.ceil(action[2])

        return device_action, bandwidth_action, waitTime

    def learn(self):
        # PPO learning process
        if self.pointer < BATCH_SIZE:  # 如果经验池中的样本不足，则不进行学习
            return

        # Sample batch from memory
        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]  # 状态
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  # 动作
        br = bt[:, -self.s_dim - 1: -self.s_dim]  # 奖励
        bs_ = bt[:, -self.s_dim:]  # 下一状态

        # 重塑奖励以确保维度正确
        br = br.reshape(-1, 1)

        # Compute advantage function
        v, v_ = self.sess.run([self.v, self.v_], {self.S: bs, self.S_: bs_})
        adv = br + GAMMA * v_ - v

        # Get current policy distribution parameters
        mu, sigma = self.sess.run([self.mu, self.sigma], {self.S: bs})
        old_pi = np.hstack([mu, sigma])

        # Multiple epochs of PPO updates (a key feature of PPO)
        for _ in range(EPOCHS):
            # Update critic network - 先更新critic网络
            _, c_loss = self.sess.run([self.ctrain, self.value_loss],
                                      {self.S: bs, self.S_: bs_, self.R: br})
            self.c_cost.append(c_loss)

            # Update actor network
            _, a_loss = self.sess.run([self.atrain, self.actor_loss],
                                      {self.S: bs, self.A: ba, self.ADV: adv, self.OLD_PI: old_pi})
            self.a_cost.append(a_loss)

    def store_transition(self, s, a, r, s_):
        # 保持与原函数相同的接口
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def save_net(self, i):
        saver = tf.train.Saver()
        now = "PPOTO"  # Changed from DDPGTO to PPOTO
        id = str(i+1)
        fname="ckpt/"+now+"_agent_"+id+".ckpt"
        save_path = saver.save(self.sess, fname)
        print("Save to path: ", save_path)

    def restore_net(self, i):
        saver = tf.train.Saver()
        now = "PPOTO"  # Changed from DDPGTO to PPOTO
        id = str(i+1)
        fname="ckpt/"+now+"_agent_"+id+".ckpt"
        saver.restore(self.sess, fname)
        print("Model restored.")
        print('Initialized')