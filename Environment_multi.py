import math
import os
import random
import sys
import networkx as nx
import pandas as pd

from EdgeDevice import EdgeDevice
from Task import Task

class MultiHopNetwork:
    def __init__(self, action_spaces, n_features, fpath, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path):
        self.map = []
        self.action_space = action_spaces
        self.n_actions = len(self.action_space)
        self.n_features = n_features
        self.task_release_time = {}
        self.schedule_path = schedule_path # schedule_path = ali_data+"device75/now_schedule.csv"
        self.network_edge_path = network_edge_path
        self.network_node_path = network_node_path
        self.device_path = device_path
        self.task_file_path = task_file_path
        self.task_pre_path = task_pre_path
        self.fpath = fpath
        self.failure_interval = 10  # One out of every 10 tasks fails
        self.task_counter = 0  # Used to track the number of tasks processed
        self.build_network()

    def build_network(self):
        # 创建空的无向图
        cecGraph = nx.Graph()
        # 读取节点文件：
        # 打开 network_node_path 文件并读取每一行。
        # 去掉换行符和回车符，然后将每行分割成一个列表。
        # 如果列表长度为3，添加一个节点到图中，节点的属性包括名字和权重
        # file_path = f"{self.fpath}network_node_info_{agent_id}.csv" # network_node_info.csv
        # f1 = open(file_path, 'r')
        # node_file_path = os.path.join(self.fpath, f"network_node_info.csv")
        node_file_path = self.network_node_path
        with open(node_file_path, 'r') as f1:
            lines = f1.readlines()
            for line in lines:
                line = line.strip()
                info = line.split(',')
                if len(info) == 3:
                    cecGraph.add_node(int(info[0]), name=info[1], weight=info[2])

        # network_edge_path 文件
        # file_path_2 = f"{self.fpath}network_edge_info.csv" # network_edge_info
        file_path_2 = self.network_edge_path
        f2 = open(file_path_2, 'r')
        lines = f2.readlines()
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            if len(info) == 3:
                cecGraph.add_edge(int(info[0]), int(info[1]), weight=float(info[2]))
                cecGraph[int(info[0])][int(info[1])]['flow'] = []
        f2.close()
        # device_path 文件
        deviceList = [] # device_path = "file/device_info.csv"
        # file_path_3 = f"{self.fpath}device_info.csv"
        file_path_3 = self.device_path
        f1 = open(file_path_3, 'r')
        lines = f1.readlines()
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            if len(info) == 4:
                device = EdgeDevice(int(info[0]), int(info[1]), float(info[2]), float(info[3]))
                deviceList.append(device)
        f1.close()

        taskList = []
        self.task_dic = {}
        f1 = open(self.task_file_path, 'r')
        lines = f1.readlines()
        # task_file_path 文件
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            task = Task(int(info[0]), int(info[1]), int(info[2]),
                        int(info[3]), int(info[4]), int(info[5]), int(info[6]))
            taskList.append(task)
            if int(info[1]) in self.task_dic.keys():
                self.task_dic[int(info[1])] = self.task_dic[int(info[1])] + 1
            else:
                self.task_dic[int(info[1])] = 1
        f1.close()
        # task_pre_path 
        f1 = open(self.task_pre_path, "r")
        lines = f1.readlines()
        edges_dic = {}
        for line in lines:
            info = line.strip("\n").split(",")
            start = int(info[1])
            end = int(info[0])
            if start != end:
                if start in edges_dic.keys():
                    tempList = edges_dic[start]
                    tempList.append(end)
                    edges_dic[start] = tempList
                else:
                    edges_dic[start] = [end]

        for key, val in edges_dic.items():
            for task in taskList:
                if task.subId == key:
                    task.setSucceList(val)
                    break
        f1.close()

        # file_path_4 = f"{self.fpath}now_schedule.csv"
        file_path_4 = self.schedule_path
        f4 = open(file_path_4, 'r')
        now_schedule = pd.read_csv(f4)
        self.cecGraph = cecGraph
        self.deviceList = deviceList
        self.state_df = now_schedule
        self.state = self.state_df.iloc[0] 
        # print(self.state)

        # schedule_path = ali_data+"device75/now_schedule.csv"
        self.taskList = taskList

    def num_agents(self):
        return self.n_features

    def getAverageWaittime(self):
        return 0

    def getAverageCtime(self):
        return 0

    def step(self, actions, tasks, t, agent_id):
        finish_times = []
        compute_times = []
        bandwidths = []
        transmit_times = []
        wait_times = []
        finished_tasks = 0
        # successful_tasks = 0

        # 用于跟踪每个主任务的子任务成功情况
        task_success_flags = {}

        for i, task in enumerate(tasks):
            self.task_counter += 1
            action = actions[i]
            task = tasks[i]
            device_action = action[0]
            bandwidth_action = action[1]
            wait_time_action = action[2]

            finish_time, compute_time, transmit_time, wait_time = self.update_state(device_action, bandwidth_action, wait_time_action, task, t)
            self.task_release_time[task.subId + task.taskId] = finish_time

            finish_times.append(finish_time)
            compute_times.append(compute_time)
            bandwidths.append(bandwidth_action)
            transmit_times.append(transmit_time)
            wait_times.append(wait_time)

            finished_tasks += 1
            if finish_time - t <= task.deadline:
                if random.random() >= self.deviceList[device_action-1].failureRate * finished_tasks:  # 100% chance of success# Assuming each task has a deadline attribute
                    task.setSuccessFlag(True)
            # if finish_time - t <= task.deadline:
            #     # Use deterministic methods to simulate node failures
            #     if self.task_counter % self.failure_interval != 0:
            #         task.setSuccessFlag(True)
                    # successful_tasks += 1
                # else: the task fails due to node failure, successful_tasks will not be increased.
            if task.taskId not in task_success_flags:
                task_success_flags[task.taskId] = True
            task_success_flags[task.taskId] &= task.successFlag
        # 计算成功的主任务数量
        successful_tasks = sum(task_success_flags.values())
        print(f"Successful tasks: {successful_tasks}")

        task_success_rate = float(successful_tasks) / len(task_success_flags) if task_success_flags else 0
        print(f"Task success rate: {task_success_rate}")
        print("task_success_flags", len(task_success_flags))
        print(f"Finished tasks: {finished_tasks}")
        overall_reward = task_success_rate  # Negative because we want to maximize success rate

        # Update the state
        self.add_new_state(t)

        # Read the updated state
        file_path = f"{self.fpath}now_schedule.csv"
        with open(file_path, 'r') as f:
            now_schedule = pd.read_csv(f)
        next_state = now_schedule.iloc[0]

        done = (finished_tasks == len(tasks))

        return next_state, overall_reward, done, task_success_rate, finish_times, compute_times, bandwidths, transmit_times, wait_times

    def add_new_state(self, t):
        a = self.state_df.iloc[0].values.tolist()
        d = pd.DataFrame(columns=self.state_df.columns)
        d.loc[-1] = a
        d['time'] = t
        self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)
        self.state_df.to_csv(f"{self.fpath}now_schedule.csv", index=0)


    def update_state(self, device_action, bandwidth_action, wait_time_action, task, t):
        wait_time = 0  # TODO: Revised by Device
        paths = self.searchGraph(self.cecGraph, task.source, device_action, task.dataSize)
        list1 = self.getEdgeList(paths)
        path_bandwidth = sys.maxsize
        bandwidth_flag = False
        max_edge_wait_time = 0

        for edge in self.cecGraph.edges:
            bandwidth_flag = True
            if edge in list1:
                edge_name = f"edge_weight_{edge[0]}{edge[1]}"
                path_bandwidth = min(self.cecGraph[edge[0]][edge[1]]['weight'], path_bandwidth)
                edge_wait_time = 0
                if t in self.state_df['time'].values:
                    edge_wait_time = self.state_df.loc[self.state_df['time'] == t, edge_name].values[0]
                max_edge_wait_time = max(max_edge_wait_time, edge_wait_time)

        if bandwidth_flag:
            if max_edge_wait_time <= wait_time_action:
                max_edge_wait_time = wait_time_action
                if path_bandwidth > bandwidth_action:
                    path_bandwidth = bandwidth_action
            else:
                if path_bandwidth > bandwidth_action:
                    path_bandwidth = bandwidth_action

        begin_t = t + max_edge_wait_time

        transmit_time = task.dataSize / path_bandwidth
        if task.source != device_action:
            for i in range(begin_t, begin_t + int(transmit_time) + 1):
                if i in self.state_df['time'].values:
                    for edge in list1:
                        edge_name = f"edge_weight_{edge[0]}{edge[1]}"
                        self.state_df.loc[self.state_df['time'] == i, edge_name] += begin_t + int(transmit_time) - i
                else:
                    self.add_new_state(i)
                    for edge in list1:
                        edge_name = f"edge_weight_{edge[0]}{edge[1]}"
                        self.state_df.loc[self.state_df['time'] == i, edge_name] = begin_t + int(transmit_time) - i

            end_transmit_time = begin_t + int(transmit_time) + 1
            if end_transmit_time not in self.state_df['time'].values:
                self.add_new_state(end_transmit_time)
                for edge in list1:
                    edge_name = f"edge_weight_{edge[0]}{edge[1]}"
                    self.state_df.loc[self.state_df['time'] == end_transmit_time, edge_name] = 0
        else:
            end_transmit_time = t

        target_device = next((device for device in self.deviceList if device.deviceId == device_action), None)
        if target_device is None:
            print(f"Error: No such device {device_action}")
            return 0, t, 0, 0, 0

        device_name = f"device_{target_device.deviceId}_time"
        p_wait_time = self.state_df.loc[self.state_df['time'] == end_transmit_time, device_name].values[0] if end_transmit_time in self.state_df['time'].values else 0

        begin_process_time = end_transmit_time + p_wait_time
        pre_list = task.getSucceList()

        # Handle the case when pre_list is empty or no matching tasks
        max_new_compute_time = 0
        if pre_list:
            matching_tasks = [self.task_release_time.get(temp_task.subId + temp_task.taskId, 0)
                              for temp_task in self.taskList if temp_task.subId in pre_list]
            if matching_tasks:
                max_new_compute_time = max(matching_tasks)

        if max_new_compute_time > begin_process_time:
            p_wait_time = max_new_compute_time - end_transmit_time

        begin_process_time = end_transmit_time + p_wait_time
        process_time = task.cload / target_device.cpuNum

        for i in range(begin_process_time, begin_process_time + int(process_time) + 1):
            if i in self.state_df['time'].values:
                self.state_df.loc[self.state_df['time'] == i, device_name] += begin_process_time + int(process_time) + 1 - i
            else:
                self.add_new_state(i)
                self.state_df.loc[self.state_df['time'] == i, device_name] = begin_process_time + int(process_time) + 1 - i

        final_time = begin_process_time + int(process_time) + 1
        if final_time not in self.state_df['time'].values:
            self.add_new_state(final_time)
            self.state_df.loc[self.state_df['time'] == final_time, device_name] = 0

        self.state_df.to_csv(f"{self.fpath}now_schedule.csv", index=0)

        wait_time = max_edge_wait_time + p_wait_time
        return final_time, process_time, transmit_time, wait_time

    def updateDeviceWaitTime(self, deviceId, waitTime):
        for device in self.deviceList:
            if deviceId == device.deviceId:
                device.setWaitTime(waitTime)

    def getNextEdgeWeight(self):
        edge_waitTime = 0
        return edge_waitTime

    def getEdgeList(self, path):
        pathList = []
        if len(path) >= 2:
            for i in range(len(path) - 1):
                if path[i] > path[i + 1]:
                    list = (path[i + 1], path[i])
                else:
                    list = (path[i], path[i + 1])
                pathList.append(list)
        elif len(path) == 1:
            pathList = [(path[0])]
        return pathList

    def searchGraph(self, graph, start, end, dataSize):
        results = []
        self.generatePath(graph, [start], end, results)
        results.sort(key=lambda x: len(x))
        if len(results) == 0:
            return []
        minPath = results[0]
        minCtime = self.calculateCtime(graph, minPath, dataSize)
        for path in results:
            tempCtime = self.calculateCtime(graph, path, dataSize)
            if tempCtime < minCtime:
                minPath = path
                minCtime = tempCtime
        return minPath

    def calculateCtime(self, graph, path, dataSize):
        sum = 0
        for i in range(len(path) - 1):
            sum += dataSize / graph.edges[path[i], path[i + 1]]['weight']
        return sum

    def generatePath(self, graph, path, end, results):
        state = path[-1]
        if state == end:
            results.append(path)
        else:
            for arc in graph[state]:
                if arc not in path:
                    self.generatePath(graph, path + [arc], end, results)

    def get_observation(self):
            return self.state

    def reset(self):
        file_path = f"file/now_schedule.csv"
        df = pd.read_csv(file_path)
        df.to_csv(f"file/now_schedule.csv", index=0)
        return self.state

