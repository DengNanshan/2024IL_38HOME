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
title = "mix_9_IP_v3_033"
conf_path = "conf/Local_IM_9.json"
loss_path = "model/loss_log/Local_"+title+"_loss.csv"
save_model_path="model/Local__"+title+".pth"
checkpoint_path = "model/Local__"+title+"_checkpoints"
epochs = 100
batch_size = 256
lr = 0.001
lr_scheduler = True
check_rate = 5

""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""
local临时测试，增加一个维度  1 agg  0 norm -1 def
IP 增加一个维度   [1-2] agg      [0-1] norm    [-1-0] def


IP9 定义了9个区域  

2---1---------0---- -1
—————————————           2
1 (a)|  2 (c) |3(b)     1
4 (a)|  5 (b) |6(c)     0
7 (c)|  8 (b) |9(a)     -1
————————————

"""
def load_data(filename):
    # print("loading data")
    # data = pd.read_csv(filename)
    # print("len of data",len(data))
    # data1 = data.sample(frac=1/9)
    # states = np.array([ast.literal_eval(state) for state in data["state"]])
    # actions = np.array([ast.literal_eval(action) for action in data["action"]])
    # print("loading data finished")


    if filename == "data/IL_data_Agg30.csv":
        data = pd.read_csv(filename)
        #
        data1 = data.sample(frac=1/9)
        states1 = np.array([ast.literal_eval(state) for state in data1["state"]])
        actions1 = np.array([ast.literal_eval(action) for action in data1["action"]])
        add1 = np.random.random((len(states1), 2))
        add1[:,0] += 1
        add1[:,1] += 1
        states1 = np.concatenate((states1, add1), axis=1)

        #
        data4 = data.sample(frac=1/9)
        states4 = np.array([ast.literal_eval(state) for state in data4["state"]])
        actions4 = np.array([ast.literal_eval(action) for action in data4["action"]])
        add4 = np.random.random((len(states4), 2))
        add4[:,1] += 1
        states4 = np.concatenate((states4, add4), axis=1)
        #
        data9 = data.sample(frac=1/9)
        states9 = np.array([ast.literal_eval(state) for state in data9["state"]])
        actions9 = np.array([ast.literal_eval(action) for action in data9["action"]])
        add9 = np.random.random((len(states9), 2))
        add9[:, 0] -= 1
        add9[:, 1] -= 1
        states9 = np.concatenate((states9, add9), axis=1)

        #全部合并
        states = np.concatenate((states1, states4, states9), axis=0)
        actions = np.concatenate((actions1, actions4, actions9), axis=0)

    elif filename == "data/IL_data_Norm30.csv":
        """
        2---1---------0---- -1
        —————————————           2
        1 (a)|  2 (c) |3(b)     1
        4 (a)|  5 (b) |6(c)     0
        7 (c)|  8 (b) |9(a)     -1
        ————————————
        
"""
        data = pd.read_csv(filename)
        #
        data3 = data.sample(frac=1 / 9)
        states3 = np.array([ast.literal_eval(state) for state in data3["state"]])
        actions3 = np.array([ast.literal_eval(action) for action in data3["action"]])
        add3 = np.random.random((len(states3), 2))
        add3[:, 0] += 1
        add3[:, 1] -= 1
        states3 = np.concatenate((states3, add3), axis=1)

        #
        data5 = data.sample(frac=1 / 9)
        states5 = np.array([ast.literal_eval(state) for state in data5["state"]])
        actions5 = np.array([ast.literal_eval(action) for action in data5["action"]])
        add5 = np.random.random((len(states5), 2))

        states5 = np.concatenate((states5, add5), axis=1)
        #
        data8 = data.sample(frac=1 / 9)
        states8 = np.array([ast.literal_eval(state) for state in data8["state"]])
        actions8 = np.array([ast.literal_eval(action) for action in data8["action"]])
        add8 = np.random.random((len(states8), 2))
        add8[:, 0] -= 1
        states8 = np.concatenate((states8, add8), axis=1)

        # 全部合并
        states = np.concatenate((states3, states5, states8), axis=0)
        actions = np.concatenate((actions3, actions5, actions8), axis=0)



    elif filename == "data/IL_data_Def30.csv":

        """ccc
             2---1---------0---- -1
             —————————————           2
             1 (a)|  2 (c) |3(b)     1
             4 (a)|  5 (b) |6(c)     0
             7 (c)|  8 (b) |9(a)     -1
             ————————————

     """
        data = pd.read_csv(filename)
        #
        data2 = data.sample(frac=1 / 9)
        states2 = np.array([ast.literal_eval(state) for state in data2["state"]])
        actions2 = np.array([ast.literal_eval(action) for action in data2["action"]])
        add2 = np.random.random((len(states2), 2))
        add2[:, 0] += 1

        states2 = np.concatenate((states2, add2), axis=1)

        #
        data6 = data.sample(frac=1 / 9)
        states6 = np.array([ast.literal_eval(state) for state in data6["state"]])
        actions6 = np.array([ast.literal_eval(action) for action in data6["action"]])
        add6 = np.random.random((len(states6), 2))
        add6[:, 1] -= 1
        states6 = np.concatenate((states6, add6), axis=1)
        #
        data7 = data.sample(frac=1 / 9)
        states7 = np.array([ast.literal_eval(state) for state in data7["state"]])
        actions7 = np.array([ast.literal_eval(action) for action in data7["action"]])
        add7 = np.random.random((len(states7), 2))
        add7[:, 0] -= 1
        states7 = np.concatenate((states7, add7), axis=1)

        # 全部合并
        states = np.concatenate((states2, states6, states7), axis=0)
        actions = np.concatenate((actions2, actions6, actions7), axis=0)

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
