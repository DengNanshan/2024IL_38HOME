"""
    用于测试模型的效果
    与v1相比修改了数据点的分布和动作空间【-5 5】

    与v2相比，使用自己定义的highway_env_random_pose ,实现随机初始化位置
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
env_conf_path = "conf/HighwayConf_Norm_Cont.json"
moedl_path = "model/ImitationModel_Norm_v3.pth"

with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

model = ImitationModel(config=conf)
if "checkpoint" in moedl_path:
    model.load_state_dict(torch.load(moedl_path)['model_state_dict'])
else:
    model.load_state_dict(torch.load(moedl_path))
# ILModel= torch.load

"""读取化配置文件"""
env = gym.make('highway-random-pose-v0', render_mode="rgb_array")
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
    print(time)
    time = time + 0.1
    # print(a)

    obs, reward, Done, T,info = env.step(a)


