""""
运行多个模型，测试模型碰撞概率，保存测试结果
"""

import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import ast
import csv
import numpy as np
import pandas as pd
from ImitationModel import ImitationModel
import torch
import json

from tools.tools import *

"""读取配置文件"""
# all_conf_path = "conf/Test_collisions_Agg.json"
# all_conf_path = "conf/Test_collisions_Agg_v3.json"
all_conf_path = "conf/Test_collisions_v3.json"

all_conf = json2dict(all_conf_path)
model_conf = json2dict(all_conf["model"])
env_confs = all_conf["environments"]
env = gym.make('highway-v0', render_mode="rgb_array")
agents_conf = all_conf["agents"] # 多个模型名字

"""循环运行多个模型并保存结果"""
for env_conf_path in env_confs:
    env_conf = json2dict(env_conf_path)
    env.configure(env_conf)
    print("load env configure:",env_conf_path)
    for agent_name in agents_conf:
        # 读取模型
        model = ImitationModel(config=model_conf)
        # 判断名字里是够是否有checkpoint
        if "checkpoint" in agent_name:
            model.load_state_dict(torch.load(agent_name)['model_state_dict'])
        else:
            model.load_state_dict(torch.load(agent_name))
        print("load model:",agent_name)
        # 测试模型
        for i in range(all_conf["run_times"]):
            Done = False
            Track = False
            time = 0
            obs, info = env.reset()
            while not Done or Track:
                on_road = env.env.env.vehicle.on_road
                if not on_road:
                    print("episode",i,"off road")
                    with open(all_conf["result_path"], 'a', newline='') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow([env_conf_path,agent_name,time,Done,Track])
                    # 记录出界次数
                    break
                # env.render()
                state = env.env.env.vehicle.d_get_state()
                s = state["state"].flatten()
                # tenor
                s = torch.Tensor(s).to(model.device)
                a = model.forward(s)
                a = a.cpu().detach().numpy()
                a = trans_model2_env(a)
                time = time + 0.1
                obs, reward, Done, Track, info = env.step(a)
                if Track or Done:
                    print("episode",i,"time",time)
                    # 写入结果
                    with open(all_conf["result_path"], 'a', newline='') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow([env_conf_path,agent_name,time,Done,Track])
                    break

#
# Done = False
# time = 0
# while not Done:
#     on_road = env.env.env.vehicle.on_road
#     if not on_road:
#         print("off road")
#         break
#     env.render()
#     state = env.env.env.vehicle.d_get_state()
#     s = state["state"].flatten()
#     # tenor
#     s = torch.Tensor(s).to(model.device)
#     a = model.forward(s)
#     a = a.cpu().detach().numpy()
#
#     a = trans_model2_env(a)
#     print(a)
#     time = time + 0.1
#     # print(a)
#
#     obs, reward, Done, T,info = env.step(a)
#
#
#
# all_a_t = np.array(all_a_t)
# print(np.mean((all_a - all_a_t) ** 2))
#
# MSELoss = np.mean((all_a - all_a_t) ** 2)
# import torch.nn as nn
# criterion = nn.MSELoss()
# a = torch.Tensor(all_a)
# a_t = torch.Tensor(all_a_t)
# loss = criterion(a, a_t)
# print("nn loss",loss)