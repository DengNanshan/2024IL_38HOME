"""
与V1差异在于
1.调整了退火策略
2 将空间范围调整到55

"""

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
title = "Agg_v3"

data_file_path = "data/IL_data_"+"Agg30"+".csv"
conf_path = "conf/ImitationModel_deep.json"
loss_path = "model/loss_log/ImitationModel_"+title+"_loss.csv"
sace_model_path="model/ImitationModel_"+title+".pth"
checkpoint_path = "model/ImitationModel_"+title+"_checkpoints"
epochs = 100
batch_size = 256 # default 256
check_rate = 1

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
states, actions = trans_data2_model(states, actions)

# load config
with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

# generate model
model = ImitationModel(config=conf)


model.train(states, actions, epochs=epochs, batch_size=batch_size,
            ir_Scheduler=True,
            lr=0.001,
            check_point_path=checkpoint_path,
            check_rate=check_rate,)
with open(loss_path, mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["count","loss","learning_rate"])
    for loss in model.losss:
        writer.writerow(loss)

# draw loss
model.draw_loss()

# save model
model.save(sace_model_path)


