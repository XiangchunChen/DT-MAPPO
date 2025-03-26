import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from others.AutoEncoder import AutoEncoder
from Environment_multi import MultiHopNetwork
from PPO import PPO
import tensorflow as tf

# # 启用PyTorch异常检测
# torch.autograd.set_detect_anomaly(True)
#
# # 检查GPU可用性（PyTorch部分）
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 检查TensorFlow GPU可用性
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

# 定义自动编码器训练过程
# def train_autoencoder(model, dataset, epochs, batch_size, learning_rate):
#     # 将模型移动到GPU
#     model = model.to(device)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(epochs):
#         for batch_data in dataloader:
#             # 将数据移动到GPU
#             bs_ba, br_bs_ = batch_data[0].to(device), batch_data[1].to(device)
#
#             optimizer.zero_grad()
#             output = model(bs_ba)
#             loss = criterion(output, br_bs_)
#             loss.backward()
#             optimizer.step()
#
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 无监督学习函数
# def unsupervised_learning(t_model, r_model, bs, ba, s_dim):
#     bs_array = bs.array[1:].to_numpy()
#     bs_array = bs_array.reshape(1, len(bs_array))
#     ba = ba.reshape(1, len(ba))
#     bs_ba = np.concatenate((bs_array, ba), axis=1)
#     bs_ba_float = bs_ba.astype(np.float32)
#
#     # 将数据移动到GPU
#     bs_ba_tensor = torch.tensor(bs_ba_float, dtype=torch.float32).to(device)
#
#     # 使用状态转移模型预测下一个状态
#     t_model.to(device)
#     bs_ = t_model.forward(bs_ba_tensor)
#     bs_ = bs_[:, -s_dim:]  # 提取预测结果的最后s_dim列
#
#     # 使用奖励模型预测奖励
#     r_model.to(device)
#     br = r_model.forward(bs_)
#
#     return br, bs_

def getDevicesByTask(deviceList, task):
    """
    返回适合给定任务的设备列表。
    如果设备ID与任务的源匹配，则认为该设备适合该任务。

    参数:
        deviceList (list): EdgeDevice对象列表
        task (Task): 要查找适合设备的任务对象

    返回:
        list: 适合该任务的设备ID列表
    """
    for device in deviceList:
        if device.deviceId == task.source:
            return device.deviceId
    return None

###### 2. 主训练循环 ######
def train(env, agents, task_num):
    # 创建TensorFlow配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 动态分配GPU内存
    config.log_device_placement = True  # 记录设备分配情况

    # 创建摘要写入器
    log_dir = "logs/"
    summary_writer = tf.summary.FileWriter(log_dir)

    # 创建指标的占位符
    completion_time_placeholder = tf.placeholder(tf.float32, shape=())
    compute_time_placeholder = tf.placeholder(tf.float32, shape=())
    success_rate_placeholder = tf.placeholder(tf.float32, shape=())
    transmit_time_placeholder = tf.placeholder(tf.float32, shape=())
    wait_time_placeholder = tf.placeholder(tf.float32, shape=())
    global_reward_placeholder = tf.placeholder(tf.float32, shape=())

    # 创建摘要操作
    completion_time_summary = tf.summary.scalar("Completion Time", completion_time_placeholder)
    compute_time_summary = tf.summary.scalar("Computation Time", compute_time_placeholder)
    success_rate_summary = tf.summary.scalar("Success Rate", success_rate_placeholder)
    transmit_time_summary = tf.summary.scalar("Transmit Time", transmit_time_placeholder)
    wait_time_summary = tf.summary.scalar("Wait Time", wait_time_placeholder)
    avg_reward_summary = tf.summary.scalar("Average Reward", global_reward_placeholder)

    with tf.Session(config=config) as sess:
        sess.run(init)
        # 将会话分配给每个代理
        for agent0 in agents:
            agent0.sess = sess

        step = 0
        completion_time_dic = {}
        global_reward_dic = {}
        computation_time_dic = {}
        transmit_time_dic = {}
        wait_time_dic = {}
        global_success_rate = {}

        devices = [i+1 for i in range(len(env.deviceList))]

        for episode_i in range(100):
            print("-------------------episode: ", episode_i, "-------------------")
            observations = env.reset()
            episode_done = False
            time_step = 1
            max_time = 100
            task_count = 0

            agent_data = {
                agent_id0: {
                    'task_time_dic': {},
                    'subtask_time_dic': {},
                    'compute_time_dic': {},
                    'transmit_time_dic': {},
                    'wait_time_dic': {},
                    'avg_reward_dic': {}
                }
                for agent_id0 in devices
            }

            while not episode_done and time_step <= max_time:
                print("------------------timestep: ", time_step, "------------------")
                tasks = getTasksByTime(env.taskList, time_step)
                task_count += len(tasks)

                if len(tasks) != 0:
                    actions = []
                    for task in tasks:
                        print("------subtask:", task.subId, "------")
                        agent_id = getDevicesByTask(env.deviceList, task)
                        agent = agents[agent_id - 1]
                        print("agent_id", agent_id)
                        observation = observations[:edges_devices_num[agent_id - 1]]  # 每个代理的状态维度不同

                        # 每个代理根据其观察选择一个动作
                        try:
                            action = agent.choose_action(observation)
                            print("subtask.source, action", task.source, action)
                            actions.append(action)
                        except ValueError as e:
                            print(f"错误: {e}")
                            # 如果出现NaN错误，使用默认动作
                            default_action = (1, 1, 1)  # 默认动作
                            print(f"使用默认动作: {default_action}")
                            actions.append(default_action)

                    # 与环境交互
                    observation_, reward, done, successful_tasks, finish_times, compute_times, bandwidths, transmit_times, wait_times = env.step(actions, tasks, time_step, agent_id)

                    print("reward", reward)
                    print("finish_times", finish_times)
                    print("compute_times", compute_times)
                    print("transmit_times", transmit_times)
                    print("wait_times", wait_times)

                    # 为每个代理存储转换
                    for i, task in enumerate(tasks):
                        agent_id = getDevicesByTask(env.deviceList, task)
                        agent = agents[agent_id - 1]

                        # 检查数据是否包含NaN
                        if np.isnan(np.array(observation)).any() or np.isnan(np.array(actions[i])).any() or np.isnan(np.array(reward)).any() or np.isnan(np.array(observation_)).any():
                            print("警告: 检测到NaN值，跳过存储此转换")
                            continue

                        agent.store_transition(observation, actions[i], reward, observation_)

                        if (step > 10) and (step % 5 == 0):
                            agent.learn()

                        agent_data[agent_id]['avg_reward_dic'][task.taskId] = max(agent_data[agent_id]['avg_reward_dic'].get(task.taskId, -float('inf')), reward)
                        agent_data[agent_id]['subtask_time_dic'][task.subId] = finish_times[i] - time_step + 1
                        agent_data[agent_id]['task_time_dic'][task.taskId] = max(agent_data[agent_id]['task_time_dic'].get(task.taskId, -float('inf')), finish_times[i] - time_step + 1)
                        agent_data[agent_id]['compute_time_dic'][task.taskId] = agent_data[agent_id]['compute_time_dic'].get(task.taskId, 0) + compute_times[i]
                        agent_data[agent_id]['transmit_time_dic'][task.taskId] = agent_data[agent_id]['transmit_time_dic'].get(task.taskId, 0) + transmit_times[i]
                        agent_data[agent_id]['wait_time_dic'][task.taskId] = agent_data[agent_id]['wait_time_dic'].get(task.taskId, 0) + wait_times[i]

                    observation = observation_
                    observations[:edges_devices_num[agent_id - 1]] = observation
                    step += 1

                    if task_count == task_num:
                        episode_done = True
                        completion_time = 0
                        compute_time = 0
                        transmit_time = 0
                        wait_time = 0
                        sum_reward = 0

                        for taskId in agent_data[agent_id]['task_time_dic'].keys():
                            completion_time += agent_data[agent_id]['task_time_dic'][taskId]
                            compute_time += agent_data[agent_id]['compute_time_dic'][taskId]
                            transmit_time += agent_data[agent_id]['transmit_time_dic'][taskId]
                            wait_time += agent_data[agent_id]['wait_time_dic'][taskId]
                            sum_reward += agent_data[agent_id]['avg_reward_dic'][taskId]

                        completion_time_dic[episode_i] = completion_time
                        computation_time_dic[episode_i] = compute_time
                        if episode_i not in global_success_rate:
                            global_success_rate[episode_i] = successful_tasks/float(task_num)
                        else:
                            global_success_rate[episode_i] += successful_tasks/float(task_num)
                        transmit_time_dic[episode_i] = transmit_time
                        wait_time_dic[episode_i] = wait_time
                        global_reward_dic[episode_i] = sum_reward/float(task_num)

                        print(f"Episode {episode_i} finished")
                        print("completion_time_dic", completion_time_dic[episode_i])
                        print("computation_time_dic", computation_time_dic[episode_i])
                        print("transmit_time_dic", transmit_time_dic[episode_i])
                        print("wait_time_dic", wait_time_dic[episode_i])
                        print("reward_dic", global_reward_dic[episode_i])
                        print("global_success_rate", global_success_rate[episode_i])

                        # 运行摘要操作并将其添加到摘要写入器
                        summaries = sess.run(
                            [completion_time_summary, compute_time_summary, success_rate_summary, transmit_time_summary, wait_time_summary, avg_reward_summary],
                            feed_dict={
                                completion_time_placeholder: completion_time_dic[episode_i],
                                compute_time_placeholder: computation_time_dic[episode_i],
                                transmit_time_placeholder: transmit_time_dic[episode_i],
                                wait_time_placeholder: wait_time_dic[episode_i],
                                global_reward_placeholder: global_reward_dic[episode_i],
                                success_rate_placeholder: global_success_rate[episode_i]  # 新增这一行
                            }
                        )

                        for summary in summaries:
                            summary_writer.add_summary(summary, episode_i)
                        break
                else:
                    env.add_new_state(time_step)
                    time_step += 1
                    if time_step > max_time:  # 添加这一行
                        episode_done = True   # 添加这一行

        # 保存模型
        for i, agent in enumerate(agents):
            agent.save_net(i)
            print(f"Saved model for agent {i+1}")

        summary_writer.flush()
        summary_writer.close()

    # 新增：将任务成功率数据保存到CSV文件
    success_rate_df = pd.DataFrame.from_dict(global_success_rate, orient='index', columns=['Task Success Rate'])
    success_rate_df.to_csv('result/task_success_rate.csv')

    return completion_time_dic, global_reward_dic, computation_time_dic, global_success_rate

###### 2. 主测试 ######
def run_model(env, agents, task_num):
    # 创建TensorFlow配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 动态分配GPU内存
    config.log_device_placement = True  # 记录设备分配情况

    with tf.Session(config=config) as sess:
        # 恢复每个代理的模型
        for i in range(len(agents)):
            agents[i].sess = sess
            agents[i].restore_net(i)

        step = 0
        completion_time_dic = {}
        reward_dic = {}
        computation_time_dic = {}
        transmit_time_dic = {}
        wait_time_dic = {}
        global_success_rate = {}

        devices = [i+1 for i in range(len(env.deviceList))]

        for episode_i in range(1):
            print("-------------------episode: ", episode_i, "-------------------")
            observations = env.reset()
            episode_done = False
            time_step = 1
            max_time = 1
            task_count = 0

            agent_data = {
                agent_id0: {
                    'task_time_dic': {},
                    'subtask_time_dic': {},
                    'compute_time_dic': {},
                    'transmit_time_dic': {},
                    'wait_time_dic': {},
                    'avg_reward_dic': {}
                }
                for agent_id0 in devices
            }

            while not episode_done or time_step <= max_time:
                print("------------------timestep: ", time_step, "------------------")
                tasks = getTasksByTime(env.taskList, time_step)
                task_count += len(tasks)

                if len(tasks) != 0:
                    actions = []
                    for task in tasks:
                        print("------subtask:", task.subId, "------")
                        agent_id = getDevicesByTask(env.deviceList, task)
                        agent = agents[agent_id - 1]
                        print("agent_id", agent_id)
                        observation = observations[:edges_devices_num[agent_id - 1]]  # 每个代理的状态维度不同

                        # 每个代理根据其观察选择一个动作
                        try:
                            action = agent.choose_action(observation)
                            print("subtask.source, action", task.source, action)
                            actions.append(action)
                        except ValueError as e:
                            print(f"错误: {e}")
                            # 如果出现NaN错误，使用默认动作
                            default_action = (1, 1, 1)  # 默认动作
                            print(f"使用默认动作: {default_action}")
                            actions.append(default_action)

                    # 与环境交互
                    observation_, local_reward, done, successful_tasks, finish_times, compute_times, bandwidths, transmit_times, wait_times = env.step(actions, tasks, time_step, agent_id)

                    print("local reward", local_reward)
                    print("finish_times", finish_times)
                    print("compute_times", compute_times)
                    print("transmit_times", transmit_times)
                    print("wait_times", wait_times)
                    print("successful_tasks", successful_tasks)

                    # Store transitions for each agent
                    for i, task in enumerate(tasks):
                        agent_id = getDevicesByTask(env.deviceList, task)
                        agent_data[agent_id]['avg_reward_dic'][task.taskId] = max(agent_data[agent_id]['avg_reward_dic'].get(task.taskId, -float('inf')), local_reward)
                        agent_data[agent_id]['subtask_time_dic'][task.subId] = finish_times[i] - time_step + 1
                        agent_data[agent_id]['task_time_dic'][task.taskId] = max(agent_data[agent_id]['task_time_dic'].get(task.taskId, -float('inf')), finish_times[i] - time_step + 1)
                        agent_data[agent_id]['compute_time_dic'][task.taskId] = agent_data[agent_id]['compute_time_dic'].get(task.taskId, 0) + compute_times[i]
                        agent_data[agent_id]['transmit_time_dic'][task.taskId] = agent_data[agent_id]['transmit_time_dic'].get(task.taskId, 0) + transmit_times[i]
                        agent_data[agent_id]['wait_time_dic'][task.taskId] = agent_data[agent_id]['wait_time_dic'].get(task.taskId, 0) + wait_times[i]

                    step += 1

                    if task_count == task_num:
                        episode_done = True

                        completion_time = 0
                        compute_time = 0
                        transmit_time = 0
                        wait_time = 0
                        sum_reward = 0

                        for agent_id in agent_data.keys():
                            for taskId in agent_data[agent_id]['task_time_dic'].keys():
                                completion_time += agent_data[agent_id]['task_time_dic'][taskId]
                                compute_time += agent_data[agent_id]['compute_time_dic'][taskId]
                                transmit_time += agent_data[agent_id]['transmit_time_dic'][taskId]
                                wait_time += agent_data[agent_id]['wait_time_dic'][taskId]
                                sum_reward += agent_data[agent_id]['avg_reward_dic'][taskId]

                        completion_time_dic[episode_i] = completion_time
                        computation_time_dic[episode_i] = compute_time
                        if episode_i not in global_success_rate:
                            global_success_rate[episode_i] = successful_tasks
                        else:
                            global_success_rate[episode_i] += successful_tasks
                        transmit_time_dic[episode_i] = transmit_time
                        wait_time_dic[episode_i] = wait_time
                        reward_dic[episode_i] = sum_reward/float(task_num)

                        print("agent_data", agent_data)
                        print(f"Episode {episode_i} finished")
                        print("completion_time_dic", completion_time_dic[episode_i])
                        print("computation_time_dic", computation_time_dic[episode_i])
                        print("transmit_time_dic", transmit_time_dic[episode_i])
                        print("wait_time_dic", wait_time_dic[episode_i])
                        print("reward_dic", reward_dic[episode_i])
                        print("global_success_rate", global_success_rate[episode_i]/float(task_num))
                else:
                    env.add_new_state(time_step)
                    time_step += 1

    return completion_time_dic, reward_dic, computation_time_dic

def getTasksByTime(taskList, time_step):
    tasks = []
    for task in taskList:
        if task.release_time == time_step:
            tasks.append(task)
    sorted(tasks, key=lambda task: task.subId)
    return tasks

def restore_state(destory_path):
    df = pd.read_csv(destory_path)
    df.to_csv("file/now_schedule.csv", index=0)

# 新增：绘制任务成功率曲线的函数
def plot_TSR(success_rate_history):
    episodes = list(success_rate_history.keys())
    success_rates = list(success_rate_history.values())

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, success_rates, marker='o')
    plt.title('Task Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Task Success Rate')
    plt.grid(True)
    plt.savefig('result/task_success_rate_plot.png')
    plt.close()

if __name__ == "__main__":
    ##################### paths & device ####################
    ali_data = "file/Task/"
    task_file_path = ali_data+"task_info_10.csv"
    task_pre_path = ali_data+"task_pre_10.csv"
    # ali_data = "rfile/Bandwidth/"
    # task_file_path = ali_data+"task_info_30.csv"
    # task_pre_path = ali_data+"task_pre_30.csv"

    fpath = "file/"
    network_node_path = "file/network_node_info.csv"
    network_edge_path = "file/network_edge_info.csv"
    # network_edge_path = ali_data+"/1/network_edge_info.csv"
    device_path = "file/device_info.csv"
    schedule_path = "file/now_schedule.csv"
    restore_path = "now_schedule.csv"

    edges_devices_num = [16, 16, 16, 16, 16, 16, 16, 16]
    devices = [1, 2, 3, 4, 5, 6, 7, 8]

    f1 = open(task_file_path, "r")
    lines = f1.readlines()
    task_num = len(lines)

    ##################### 1. initialize ####################
    # 1.1 initial env and agent
    restore_state(restore_path)
    env = MultiHopNetwork(devices, edges_devices_num[0], fpath, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path)

    # 1.2 get state & action name
    s_dim = env.n_features  # 148
    a_dim = 3  # < device, bandwidth, waitTime>
    a_bound = env.n_actions

    print(f"s_dim={s_dim}, a_dim={a_dim}, a_bound={a_bound}")

    # 1.3 multi-agent initialization
    agents = []
    NUM_AGENT = 8  # TODO Revised by the number of devices
    for i in range(1, NUM_AGENT+1):
        s_dim = edges_devices_num[i-1]
        # print(s_dim)
        agent = PPO(a_dim, s_dim, a_bound, scope=f'agent_{i}')
        agents.append(agent)
        print(f"Initializing agent {i}.....")

    init = tf.global_variables_initializer()  # 在创建会话后，立即初始化所有变量

    input_dim = s_dim - 1 + a_dim  # 150
    hidden_dim = 128
    output_dim = s_dim - 1  # 147
    transaction_model = AutoEncoder(input_dim, hidden_dim, output_dim)  # 150 147

    r_input_dim = s_dim - 1
    r_output_dim = 1
    reward_model = AutoEncoder(r_input_dim, hidden_dim, r_output_dim)  # 147 1

    ##################### 2. training ####################
    completion_time_dic, global_reward_dic, computation_time_dic, global_success_rate = train(env, agents, task_num)

    # 新增：调用绘图函数
    plot_TSR(global_reward_dic)
    # 2. evaluation
    # run_model(env, agents, task_num)

