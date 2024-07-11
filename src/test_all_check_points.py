"""
测试包含所有checkpoints的模型
"""

import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import ast
import csv

# load data
import numpy as np
import pandas as pd
from ImitationModel import ImitationModel
import torch
import json
from tools.tools import *
conf_path = "conf/ImitationModel_deep.json"
env_conf_path = "conf/HighwayConf_Def_cont.json"
moedl_path = "model/ImitationModel_Agg_e100_b128_deep_range55_ExponentialLR2.pth"

with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

model = ImitationModel(config=conf)
model.load_state_dict(torch.load(moedl_path))
# ILModel= torch.load

"""读取化配置文件"""
env = gym.make('highway-v0', render_mode="rgb_array")
import json
with open(env_conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)
env.configure(conf)
obs, info = env.reset()
Done = False
time = 0
while not Done:
    on_road = env.env.env.vehicle.on_road
    if not on_road:
        print("off road")
        break
    env.render()
    state = env.env.env.vehicle.d_get_state()
    s = state["state"].flatten()
    # tenor
    s = torch.Tensor(s).to(model.device)
    a = model.forward(s)
    a = a.cpu().detach().numpy()

    a = trans_model2_env(a)
    print(a)
    time = time + 0.1
    # print(a)

    obs, reward, Done, T,info = env.step(a)

# for vehicle in env.env.road.vehicles[1:]:
#     state = vehicle.d_get_state()
#     s = state["state"].flatten()
#     a = [state["action"]["steering"], state["action"]["acceleration"]]
#     # 写入文件
#     # writer.writerow([str(s),str(a)])
#     writer.writerow([','.join(map(str, s)), ','.join(map(str, a))])


