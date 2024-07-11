"""
验证模型的效果

1、 验证数据采用数据集里的数据

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




conf_path = "conf/ImitationModel_deep.json"
# moedel_path = "model/ImitationModel_Agg_e100_b128_deep.pth"
moedel_path = "model/ImitationModel_Agg_e300_ExponentialLR2.pth"

""" 读取数据"""
title = "Agg_e100_b128_deep"
data_file_path = "data/IL_data_"+"Agg30"+".csv"
def load_data(filename):
    print("loading data")
    data = pd.read_csv(filename)
    # string to list
    states = np.array([ast.literal_eval(state) for state in data["state"]])
    actions = np.array([ast.literal_eval(action) for action in data["action"]])
    print("loading data finished")
    return states, actions

def load_muti_file(filenamepath):
    all_state = []
    all_action = []
    for file_name in filenamepath:
        states,actions = load_data(file_name)
        all_state.append(states)
        all_action.append(actions)
    combined_states = np.concatenate(all_state, axis=0)
    combined_actions = np.concatenate(all_action, axis=0)

    return combined_states, combined_actions

states, actions = load_data(data_file_path)
""" 读取数据完成"""



"""读取模型"""
with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

model = ImitationModel(config=conf)
model.load_state_dict(torch.load(moedel_path))

"""读取模型完成"""


"""验证模型的效果"""
all_a_t=model.forward(torch.Tensor(states).to(model.device))

all_a = actions
# t = all_a[:, 0].copy()
# all_a[:, 0] = all_a[:, 1]
# all_a[:, 1] = t

all_a = torch.Tensor(all_a).to(model.device)
""""""
import torch.nn as nn
criterion = nn.MSELoss()

loss = criterion(all_a, all_a_t)
print("loss",loss)


