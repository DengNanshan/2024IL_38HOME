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
import matplotlib.pyplot as plt

file_path = "data/IL_data5.csv"


def load_data(filename):
    print("loading data")
    data = pd.read_csv(filename)
    states = np.array([ast.literal_eval(state) for state in data["state"]])
    actions = np.array([ast.literal_eval(action) for action in data["action"]])
    print("loading data finished")
    return states, actions
states, actions = load_data(file_path)

# train model
model = ImitationModel(28, 2)
model.train(states, actions, epochs=10, batch_size=64)
with open("model/ImitationModel_5_loss.csv", mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["count","loss"])
    for loss in model.losss:
        writer.writerow(loss)
model.draw_loss()
model.save("model/ImitationModel_5.pth")


