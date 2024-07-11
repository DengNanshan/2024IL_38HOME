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
from tools.tools import *



conf_path = "conf/ImitationModel_deep.json"
# moedel_path = "model/ImitationModel_Agg_e100_b128_deep.pth"
moedel_path = "model/ImitationModel_Agg_e300_ExponentialLR2.pth"
moedel_path = "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_10.pth"

moedel_paths = ["model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_0.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_10.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_20.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_30.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_40.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_50.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_60.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_70.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_80.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2_checkpoints/ImitationModel_90.pth",
                "model/ImitationModel_Agg_e300_ExponentialLR2.pth"]
moedel_paths = ["model/ImitationModel_Agg_test_checkpoints/ImitationModel_0.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_1.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_2.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_3.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_4.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_5.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_6.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_7.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_8.pth",
                "model/ImitationModel_Agg_test_checkpoints/ImitationModel_9.pth",
                "model/ImitationModel_Agg_test.pth"]



""" 读取数据"""
data_file_path = "data/IL_data_"+"Agg30"+".csv"
def load_data(filename):
    print("loading data")
    """
    
    TEMP [0:10]
    
    """
    data = pd.read_csv(filename)[0:20000]
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
states, actions = trans_data2_model(states, actions)


"""读取模型"""
with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

model = ImitationModel(config=conf)

for agent in moedel_paths:
    print(data_file_path,agent)
    if "checkpoint" in agent:
        model.load_state_dict(torch.load(agent)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(agent))
    """验证模型的效果"""
    all_a_t=model.forward(torch.Tensor(states).to(model.device))
    all_a = actions
    all_a = torch.Tensor(all_a).to(model.device)
    """"""
    import torch.nn as nn
    criterion = nn.MSELoss()

    loss = criterion(all_a, all_a_t)
    print("loss",loss)

