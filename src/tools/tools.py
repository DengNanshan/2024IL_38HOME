

import torch.nn as nn
import pandas as pd
import ast
import numpy as np


def creat_mlp(input,
              layers,
              output,
              activation_fn = nn.ReLU):
    if len(layers)>0:
        modules = [nn.Linear(input,layers[0]),activation_fn]
    else:
        modules=[]

    for i in range (len(layers)-1):
        modules.append(nn.Linear(layers[i],layers[i+1]))
        modules.append(activation_fn)

    modules.append(nn.Linear(layers[-1],output))

    return modules

def trans_data2_model(states, actions):
    """action: [steering, acceleration] change to [acceleration, steering]"""
    actions = np.array(actions)
    t = actions[:, 0].copy()
    actions[:, 0] = actions[:, 1]
    actions[:, 1] = t


    """steering range form -pi/2 - pi/2 to -5 - 5"""
    actions[:, 1] = actions[:, 1] * 10/np.pi
    return states, actions

def trans_model2_env(actions):
    """steering range form -5 - 5 to -pi/2 - pi/2"""
    # actions[:, 1] = actions[:, 1] * np.pi/10
    actions[1] = actions[1] * np.pi/10
    return actions
def creat_mlp(input,
              layers,
              output,
              activation_fn = nn.ReLU):
    if len(layers)>0:
        modules = [nn.Linear(input,layers[0]),activation_fn]
    else:
        modules=[]

    for i in range (len(layers)-1):
        modules.append(nn.Linear(layers[i],layers[i+1]))
        modules.append(activation_fn)

    modules.append(nn.Linear(layers[-1],output))

    return modules

def load_data(filename):
    print("loading data")
    data = pd.read_csv(filename)
    states = np.array([ast.literal_eval(state) for state in data["state"]])
    actions = np.array([ast.literal_eval(action) for action in data["action"]])
    print("loading data finished")
    return states, actions

import json
def json2dict(filename):
    with open(filename) as f:
        conf_str = f.read()
    return json.loads(conf_str)