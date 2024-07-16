import sys
sys.path.append('..')
import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import ast
import csv
import json
# load data
import numpy as np
import pandas as pd
from ImitationModel import ImitationModel
import torch
import matplotlib.pyplot as plt
from tools.tools import trans_data2_model
""""""""""""""""""""""""""""""""""""""""""
data_files = [
    "data/IL_data_Agg30.csv",
    "data/IL_data_Def30.csv",
    "data/IL_data_Norm30.csv"
]
title = "mix_Local_v3_full"
conf_path = "conf/Local_IM.json"
loss_path = "model/loss_log/Local_"+title+"_loss.csv"
save_model_path="model/Local__"+title+".pth"
checkpoint_path = "model/Local__"+title+"_checkpoints"
epochs = 100
batch_size = 256
lr = 0.001
lr_scheduler = True
check_rate = 5

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 测试cuda
if torch.cuda.is_available():
    print("cuda is available")

"""
local临时测试，增加一个维度  1 agg  0 norm -1 def
"""
def load_data(filename):
    print("loading data")
    data = pd.read_csv(filename)
    print("len of data", len(data))
    # string to list
    states = np.array([ast.literal_eval(state) for state in data["state"]])
    actions = np.array([ast.literal_eval(action) for action in data["action"]])
    print("loading data finished")
    """"增加一个维度"""
    if filename == "data/IL_data_Agg30.csv":
        add1 = np.ones((len(states), 1))
        states = np.concatenate((states, add1), axis=1)
    elif filename == "data/IL_data_Norm30.csv":
        add0 = np.zeros((len(states), 1))
        states = np.concatenate((states, add0), axis=1)
    elif filename == "data/IL_data_Def30.csv":
        add_1 = -np.ones((len(states), 1))
        states = np.concatenate((states, add_1), axis=1)

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
    print("Load all files finished")
    return combined_states, combined_actions

states, actions = load_muti_file(data_files)

"""
states 增加一维
"""


""""""""""""""

states, actions = trans_data2_model(states, actions)


# load config
with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

# generate model
model = ImitationModel(config=conf)


model.train(states, actions, epochs=epochs, batch_size=batch_size,
            check_point_path=checkpoint_path,
            lr=lr,
            ir_Scheduler=lr_scheduler,
            check_rate=check_rate)
with open(loss_path, mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["count","loss","learning_rate"])
    for loss in model.losss:
        writer.writerow(loss)

# draw loss
model.draw_loss()

# save model
model.save(save_model_path)

#
