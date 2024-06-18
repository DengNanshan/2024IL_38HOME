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

model = ImitationModel(28, 2)
model.load_state_dict(torch.load("model/ImitationModel_5.pth"))
# ILModel= torch.load

"""读取配置文件"""
env = gym.make('highway-v0', render_mode="rgb_array")
import json
with open("conf/HighwayConf_test.json", "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)
env.configure(conf)
obs, info = env.reset()

Done = False
while not Done:
    env.render()
    state = env.env.env.vehicle.d_get_state()
    s = state["state"].flatten()
    # tenor
    s = torch.Tensor(s).to(model.device)
    a = model.forward(s)
    a = a.cpu().detach().numpy()
    obs, reward, Done, T,info = env.step(a)
    print("time",T)


