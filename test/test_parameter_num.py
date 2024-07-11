"""
获取模型参数量
"""

import numpy as np
import pandas as pd
import torch
import json
import random
import sys
import copy
sys.path.append('..//src')
from ImitationModel import ImitationModel

conf = {
  "input_size": 5,
  "output_size": 5,
  "hidden_size": [512,512,512]
}
# generate model
model = ImitationModel(config=conf)
print(model.parameters())
print(model.state_dict())
# 获取模型参数量
def get_parameter_num(model):
    return sum(p.numel() for p in model.parameters())

print(get_parameter_num(model))