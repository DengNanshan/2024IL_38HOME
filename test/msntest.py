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
import torch.nn as nn
import torch

a = torch.tensor([1., 2., 3., 4.])
b = a+4
criterion = nn.MSELoss()

loss = criterion(a, b)
print("loss", loss)

