import math
import sys
from random import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from AutoEncoder import AutoEncoder
# from Environment import MultiHopNetwork
# from DDPG import DDPG

from AutoEncoder import AutoEncoder
from Environment_multi import MultiHopNetwork
from PPO import PPO


# from simple_maddpg.maddpg_agent import Agent


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorflow as tf
# import wandb

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")


# HYPER PARAMETERS IN SIMPLE MADDPG
# NUM_EPISODE = 1000
# NUM_STEP = 100
# MEMORY_SIZE = 10000
# BATCH_SIZE = 512
# TARGET_UPDATE_INTERVAL = 200

# LR_ACTOR = 0.001
# LR_CRITIC = 0.001
# HIDDEN_DIM = 64
# GAMMA = 0.99
# TAU = 0.01

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
NUM_STEP = 100
NUM_EPISODE = 1


#####################  paths & device ####################
ali_data = "file/Test/Alibaba_test_data/"
task_file_path = ali_data+"task_info_40.csv"
task_pre_path = ali_data+"task_pre_40.csv"
fpath = "file/"
# network_node_path = ali_data+"device75/graph_node.csv"
# network_edge_path = ali_data+"device75/graph_edge.csv"
# device_path = ali_data+"device75/device_info.csv"
# schedule_path = ali_data+"device75/now_schedule.csv"
# destory_path = ali_data+"device75/restore_schedule.csv"


# network_node_path = ali_data + "device25/graph_node.csv"
# network_edge_path = ali_data + "device25/graph_edge.csv"
# device_path = ali_data + "device25/device_info.csv"
# schedule_path = ali_data + "device25/now_schedule.csv"
# # DQN/file/device25
# edges_devices_num = 50
# devices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

network_node_path = "../file/network_node_info.csv"
network_edge_path = "../file/network_edge_info.csv"
device_path = "../file/device_info.csv"
schedule_path = "../file/now_schedule.csv"
# edges_devices_num = [4, 6, 6, 4]
edges_devices_num = [16, 16, 16, 16,16, 16, 16, 16]
devices = [1,2,3,4,5,6,7,8]


f1 = open(task_file_path, "r")
lines = f1.readlines()
task_num = len(lines)

# destory(destory_path)
dic_task_time = {}

# Define the training process
def train_autoencoder(model, dataset, epochs, batch_size, learning_rate):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_data in dataloader:
            bs_ba, br_bs_ = batch_data
            optimizer.zero_grad()
            output = model(bs_ba)
            loss = criterion(output, br_bs_)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Return: The reward at the next moment (the reward is related to the status)
def unsupervised_learning(t_model, r_model, bs, ba, s_dim):
    bs_array = bs.array[1:].to_numpy()
    bs_array = bs_array.reshape(1, len(bs_array))
    ba = ba.reshape(1, len(ba))
    bs_ba = np.concatenate((bs_array, ba), axis=1)
    bs_ba_float = bs_ba.astype(np.float32)
    # print("bs_ba.dtype:", bs_ba.dtype)
    bs_ba_tensor = torch.tensor(bs_ba_float, dtype=torch.float32)
    bs_ = t_model.forward(bs_ba_tensor) # 使用状态转移模型 t_model 预测下一个状态 bs_
    bs_ = bs_[:, -s_dim:] # 提取预测结果的最后 s_dim 列，假设这部分是下一个状态的表示。
    br = r_model.forward(bs_) # 使用奖励模型 r_model 预测给定下一个状态 bs_ 的奖励 br
    return br, bs_

# Function: Check whether the specified node node_num in the state s at the corresponding time has the resources required by the task t.
# Output: Output the required waiting time.

def check_node_state(device, s, task):
    if device.resource > task.resource:
        wait_time = s["device_"+str(device.deviceId)+"_time"]
        return wait_time
    else:
        return sys.maxsize

def getDevicesByTask(deviceList, task):
    """
    Returns a list of devices that are suitable for the given task.
    A device is considered suitable if its ID matches the task's source.

    Parameters:
    deviceList (list): A list of EdgeDevice objects
    task (Task): The task object to find suitable devices for

    Returns:
    list: A list of device IDs that are suitable for the task
    """
    for device in deviceList:
        if device.deviceId == task.source:
            return device.deviceId

    return None

def checkAllocated(taskList):
    res = False
    for task in taskList:
        if not task.isAllocated:
            res = True
    return res

def getTasksByTime(taskList, time_step):
    tasks = []
    for task in taskList:
        if task.release_time == time_step:
            tasks.append(task)
    sorted(tasks, key=lambda task: task.subId)
    return tasks

def destory(destory_path):
    df = pd.read_csv(destory_path)
    df.to_csv("file/now_schedule.csv", index=0)

def plotCompletionTime(completion_time_dic,name):
    f1 = open("result/"+name+".csv", "w")
    x = []
    y = []
    for key, value in completion_time_dic.items():
        f1.write(str(key)+","+str(value)+"\n")
        x.append(key)
        y.append(value)
    f1.close()
    plt.plot(x, y)
    plt.ylabel(name)
    plt.xlabel('Episodes')
    plt.savefig("result/"+name+'.pdf')
    plt.show()


def run_model(env, agents, task_num):

    for i in range(len(agents)):
        agents[i].restore_net(i)  # Restore each agent's model
    step = 0
    completion_time_dic = {}
    reward_dic = {}
    computation_time_dic = {}

    all_reward = np.zeros(len(devices) + 1)
    for episode_i in range(10000):
        print("-------------------episode: ", episode_i, "-------------------")
        multi_obs = []
        multi_done = []
        for id in devices:
            agent_obs = env.reset(id) # type = <class 'pandas.core.series.Series'>
            multi_obs.append(agent_obs)
            multi_done.append(False) # multi_done = [False, False, False, False]

        episode_done = False
        time_step = 1
        max_time = 10
        task_count = 0
        agent_data = {
            agent_id: {
                'task_time_dic': {},
                'subtask_time_dic': {},
                'compute_time_dic': {},
                'transmit_time_dic': {},
                'avg_reward_dic': {}
            }
            for agent_id in devices
        }

        while not episode_done and time_step < max_time:
            print(f"-------------------timestep: {time_step}-------------------")
            tasks = getTasksByTime(env.taskList, time_step)
            task_count += len(tasks)

            if len(tasks) != 0:
                for task in tasks:
                    print(f"task: {task.subId}")
                    agent_id = getDevicesByTask(env.deviceList, task)

                    print("-----------------agent_id: ", agent_id, "-----------------")
                    agent = agents[agent_id - 1]
                    single_obs = multi_obs[agent_id - 1]
                    print(f"single_obs = {single_obs}, type = {type(single_obs)}")

                    single_obs = single_obs[:edges_devices_num[agent_id - 1]] # The state dimension of each agent is different
                    print(f"single_obs = {single_obs}, type = {type(single_obs)}")
                    print(f"Dimension of single_obs: {single_obs.shape if hasattr(single_obs, 'shape') else len(single_obs)}")
                    # Each agent chooses an action based on its observation
                    action = agent.choose_action(single_obs)

                    # interact with env
                    observation_, reward, done, finishTime, computeTime, bandwidth, transmitTime = env.step(action, task, time_step, agent_id)

                    if task.taskId in agent_data[agent_id]['avg_reward_dic']:
                        agent_data[agent_id]['avg_reward_dic'][task.taskId] = max(agent_data[agent_id]['avg_reward_dic'][task.taskId], reward)
                    else:
                        agent_data[agent_id]['avg_reward_dic'][task.taskId] = reward

                    agent_data[agent_id]['subtask_time_dic'][task.subId] = finishTime-time_step+1

                    if task.taskId in agent_data[agent_id]['task_time_dic']:
                        agent_data[agent_id]['task_time_dic'][task.taskId] = max(agent_data[agent_id]['task_time_dic'][task.taskId], finishTime-time_step+1)
                    else:
                        agent_data[agent_id]['task_time_dic'][task.taskId] = finishTime-time_step+1

                    if task.taskId in agent_data[agent_id]['compute_time_dic']:
                        agent_data[agent_id]['compute_time_dic'][task.taskId] += computeTime
                    else:
                        agent_data[agent_id]['compute_time_dic'][task.taskId] = computeTime

                    if task.taskId in agent_data[agent_id]['transmit_time_dic']:
                        agent_data[agent_id]['transmit_time_dic'][task.taskId] += transmitTime
                    else:
                        agent_data[agent_id]['transmit_time_dic'][task.taskId] = transmitTime

            else:
                env.add_new_state(time_step)

            print(task_count)
            if task_count == task_num:
                episode_done = True
                completion_time = 0
                compute_time = 0
                success_task_num = 0
                deadline = 67 #TODO: set each task with deadline
                for taskId in agent_data[agent_id]['task_time_dic'].keys():
                    completion_time += agent_data[agent_id]['task_time_dic'][taskId]
                    if agent_data[agent_id]['task_time_dic'][taskId] <= deadline:
                        success_task_num += 1

                node_failure_num = 0 #TODO: set each node with failure_num
                for agent_id in devices:
                    node_failure_num += devices[agent_id].failure_num

                task_success_rate = (success_task_num-node_failure_num) / task_num


                print("task_success_rate:", task_success_rate)
                # print("subtask_time_dic")
                # print(agent_data[agent_id]['subtask_time_dic'])
                completion_time_dic[episode_i] = completion_time
                sum_reward = 0
                for task, reward in agent_data[agent_id]['avg_reward_dic'].items():
                    sum_reward += reward
                print("completion_time:", completion_time)
                if sum_reward > all_reward[agent_id]:
                    all_reward[agent_id] = sum_reward
                print("reward:", all_reward[agent_id])
                # break

        if np.any(all_reward):
            avg_reward = sum(all_reward) / len(all_reward)
            print(f"Average reward at timestep {time_step}: {avg_reward}")
        time_step += 1
        # agent.save_net()
        step += 1

if __name__ == "__main__":

    #####################  1. initialize ####################

    # 1.1 initial env and agent
    env = MultiHopNetwork(devices, edges_devices_num[0], fpath, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path)

    # 1.2 get state & action name
    s_dim = env.n_features # 148
    
    a_dim = 3  # < device, bandwidth, waitTime>
    a_bound = env.n_actions
    print(f"s_dim={s_dim}, a_dim={a_dim}, a_bound={a_bound}")

    # 1.3 multi-agent initialization
    agents = []
    for i in devices:
        s_dim = edges_devices_num[i-1]
        # print(s_dim)
        agent = DDPG(a_dim, s_dim, a_bound, scope=f'agent_{i}') 
        agents.append(agent)
        print(f"Initializing agent {i}.....")
    init = tf.global_variables_initializer() # 在创建会话后，立即初始化所有变量

    input_dim = s_dim - 1 + a_dim # 150
    hidden_dim = 128
    output_dim = s_dim - 1 # 147
    transaction_model = AutoEncoder(input_dim, hidden_dim, output_dim) # 150 147
    r_input_dim = s_dim - 1 
    r_output_dim = 1
    reward_model = AutoEncoder(r_input_dim, hidden_dim, r_output_dim) # 147 1

    #####################  2. training ####################

    # completion_time_dic, reward_dic, computation_time_dic = run_model(env, agents, task_num)
    # min_sum = sys.maxsize
    # plotCompletionTime(completion_time_dic, "completion_time")
    # plotCompletionTime(reward_dic, "reward")
    # for i in devices:
    #     agents[i].plot_cost()


    # 2. evaluation
    run_model(env, agents, task_num)
