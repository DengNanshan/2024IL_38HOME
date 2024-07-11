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

""""""""""""""""""""""""""""""""""""""""""
data_files = [
    "data/IL_data_Agg30.csv",
    "data/IL_data_Def30.csv",
    "data/IL_data_Norm30.csv"
]

conf_path = "conf/ImitationModel_deep.json"
loss_path = "model/loss_log/ImitationModel_Mix_e100_b128_loss.csv"
save_model_path="model/ImitationModel_Mix_e100_b128.pth"
epochs = 100
batch_size = 256
lr = 0.01
lr_scheduler = True

""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
    print("Load all files finished")
    return combined_states, combined_actions

states, actions = load_muti_file(data_files)


# load config
with open(conf_path, "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)

# generate model
model = ImitationModel(config=conf)


model.train(states, actions, epochs=epochs, batch_size=batch_size,lr=lr,ir_Scheduler=lr_scheduler)
with open(loss_path, mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["count","loss"])
    for loss in model.losss:
        writer.writerow(loss)

# draw loss
model.draw_loss()

# save model
model.save(save_model_path)


