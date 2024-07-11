
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

conf_path = "conf/ImitationModel_deep.json"
env_conf_path = "conf/HighwayConf_Agg_test.json"
moedel_path = "model/ImitationModel_Agg_e100_b128_deep.pth"

with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

model = ImitationModel(config=conf)
model.load_state_dict(torch.load(moedel_path))
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

all_a = []
all_a_t = []
while not Done:
    on_road = env.env.env.vehicle.on_road
    if not on_road:
        print("off road")
        break
    env.render()
    """
    获取环境车的动作与模型动作做对比
    """
    for vehicle in env.env.road.vehicles[1:]:
        state = vehicle.d_get_state()
        s = state["state"].flatten()
        a = [state["action"]["steering"], state["action"]["acceleration"]]
        #
        s_t = torch.Tensor(s).to(model.device)
        a_t = model.forward(s_t)
        a_t = a_t.cpu().detach().numpy()
        # print("a", a,a_t)
        all_a.append(a)
        all_a_t.append(a_t)
        # 打印并且对整齐a与a_t小数点

    """固定动作"""
    obs, reward, Done, Track, info = env.step(4)
    if Track:
        print("Track")
        break


# 对比all_a 与all_a_t的MSELoss
all_a = np.array(all_a)
t = all_a[:, 0].copy()
all_a[:, 0] = all_a[:, 1]
all_a[:, 1] = t


all_a_t = np.array(all_a_t)
print(np.mean((all_a - all_a_t) ** 2))

MSELoss = np.mean((all_a - all_a_t) ** 2)
import torch.nn as nn
criterion = nn.MSELoss()
a = torch.Tensor(all_a)
a_t = torch.Tensor(all_a_t)
loss = criterion(a, a_t)
print("nn loss",loss)


