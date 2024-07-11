"""
生成一个随机数据集合，用ImitationLearning模型进行训练，测试参数对模型的影响
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


random.seed(0)

# 生成数据

states = torch.randn(10900, 5)
actions = copy.deepcopy(states)



conf = {
  "input_size": 5,
  "output_size": 5,
  "hidden_size": [512,512,512]
}
# generate model
model = ImitationModel(config=conf)


# train model
model.train(states, actions,batch_size=64,epochs=1000,ir_Scheduler=True,
            loss_show=True)

# draw loss
model.draw_loss()