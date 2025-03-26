import math
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import logging

np.random.seed(1)

# 检查GPU是否可用
print("GPU是否可用:", tf.test.is_gpu_available())

print("TensorFlow GPU是否可用:", tf.test.is_gpu_available())
from tensorflow.python.client import device_lib
print("TensorFlow可用的物理设备:", [device.name for device in device_lib.list_local_devices() if device.device_type == 'GPU'])

# print("TensorFlow可用的物理设备:", tf.config.list_physical_devices('GPU'))

# 设置TensorFlow使用GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已设置TensorFlow GPU内存增长")
    except RuntimeError as e:
        print("设置TensorFlow GPU内存增长时出错:", e)

# 设置使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 动态分配GPU内存
config.log_device_placement = True  # 记录设备分配情况


##################### 超参数 ####################
MAX_EPISODES = 1000
LR_A = 0.0003  # actor的学习率
LR_C = 0.0003  # critic的学习率
GAMMA = 0.99   # 折扣因子
EPSILON = 0.2  # 裁剪参数
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
OUTPUT_GRAPH = False
UPDATE_STEPS = 10  # PPO更新步数
EPOCHS = 10  # PPO训练轮数

############################### PPO ####################################

class PPO(object):
    def __init__(self, a_dim, s_dim, a_bound, scope):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 动态分配GPU内存
        config.log_device_placement = True  # 记录设备分配情况
        self.sess = tf.Session(config=config)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.ADV = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.OLD_PI = tf.placeholder(tf.float32, [None, a_dim * 2], 'old_pi')

        with tf.variable_scope(scope):
            with tf.device('/device:GPU:0'):
                with tf.variable_scope('Actor'):
                    self.mu, self.sigma = self._build_a(self.S, scope='eval', trainable=True)
                    self.old_mu, self.old_sigma = tf.split(self.OLD_PI, num_or_size_splits=2, axis=1)
                    self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
                    self.old_normal_dist = tf.distributions.Normal(self.old_mu, self.old_sigma)
                    self.a = self.normal_dist.sample()
                    self.a = tf.clip_by_value(self.a, -self.a_bound, self.a_bound)

                with tf.variable_scope('Critic'):
                    self.v = self._build_c(self.S, scope='eval', trainable=True)
                    self.v_ = self._build_c(self.S_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/target')

        self.soft_replace = [tf.assign(t, t) for t in self.ct_params]

        with tf.variable_scope('ppo_loss'):
            self.v_target = self.R + GAMMA * self.v_
            self.value_loss = tf.reduce_mean(tf.square(self.v_target - self.v))
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.value_loss, var_list=self.ce_params)

            with tf.variable_scope('surrogate'):
                ratio = tf.exp(self.normal_dist.log_prob(self.A) - self.old_normal_dist.log_prob(self.A))
                ratio = tf.reduce_mean(ratio, axis=1, keepdims=True)
                surr = ratio * self.ADV
                clipped_surr = tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * self.ADV
                self.policy_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))
                self.entropy = tf.reduce_mean(self.normal_dist.entropy())
                self.actor_loss = self.policy_loss - 0.01 * self.entropy

            optimizer = tf.train.AdamOptimizer(LR_A)
            grads_and_vars = optimizer.compute_gradients(self.actor_loss, var_list=self.ae_params)
            capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars if grad is not None]
            self.atrain = optimizer.apply_gradients(capped_grads_and_vars)

        self.a_cost = []
        self.c_cost = []

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)

            mu = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh,
                                 kernel_initializer=tf.initializers.variance_scaling(),
                                 name='mu', trainable=trainable)
            mu = tf.multiply(mu, self.a_bound, name='scaled_mu')

            log_sigma = tf.layers.dense(net, self.a_dim, activation=None, name='log_sigma', trainable=trainable)
            sigma = tf.exp(log_sigma) + 0.1

            return mu, sigma

    def _build_c(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l5', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)

    def choose_action(self, s):
        mu, sigma = self.sess.run([self.mu, self.sigma], {self.S: s[np.newaxis, :]})

        if np.isnan(mu).any() or np.isnan(sigma).any():
            print("警告: 检测到NaN值 - mu:", mu, "sigma:", sigma)
            mu = np.nan_to_num(mu, nan=1.0)
            sigma = np.nan_to_num(sigma, nan=0.1)

        action = np.random.normal(mu[0], sigma[0])
        action = np.clip(action, 1, self.a_bound)

        if np.isnan(action).any():
            print("警告: 动作包含NaN值，使用默认值")
            action = np.ones_like(action)

        device_action = math.ceil(action[0])
        bandwidth_action = math.ceil(action[1])
        waitTime = math.ceil(action[2])

        return device_action, bandwidth_action, waitTime

    def learn(self):
        if self.pointer < BATCH_SIZE:
            return

        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim].reshape(-1, 1)
        bs_ = bt[:, -self.s_dim:]

        v, v_ = self.sess.run([self.v, self.v_], {self.S: bs, self.S_: bs_})
        adv = br + GAMMA * v_ - v

        mu, sigma = self.sess.run([self.mu, self.sigma], {self.S: bs})
        old_pi = np.hstack([mu, sigma])

        for _ in range(EPOCHS):
            _, c_loss = self.sess.run([self.ctrain, self.value_loss],
                                      {self.S: bs, self.S_: bs_, self.R: br})
            self.c_cost.append(c_loss)

            _, a_loss = self.sess.run([self.atrain, self.actor_loss],
                                      {self.S: bs, self.A: ba, self.ADV: adv, self.OLD_PI: old_pi})
            self.a_cost.append(a_loss)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def save_net(self, i):
        saver = tf.train.Saver()
        now = "PPOTO"
        id = str(i+1)
        fname="ckpt/"+now+"_agent_"+id+".ckpt"
        save_path = saver.save(self.sess, fname)
        print("Save to path: ", save_path)

    def restore_net(self, i):
        saver = tf.train.Saver()
        now = "PPOTO"
        id = str(i+1)
        fname="ckpt/"+now+"_agent_"+id+".ckpt"
        saver.restore(self.sess, fname)
        print("Model restored.")

print('Initialized')