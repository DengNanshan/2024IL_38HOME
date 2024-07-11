
# 读取txt文件 读取其中学习率的变化

import os
import numpy as np
import matplotlib.pyplot as plt
import re

file = "learningrate.txt"
with open(file, "r") as f:
    lines = f.readlines()
    learning_rate = []
    for line in lines:
        if "Learning Rate" in line:
            print(re.findall(r"\d+\.\d+", line))
            if len(re.findall(r"\d+\.\d+", line)) == 0:
                continue
            lr = float(re.findall(r"\d+\.\d+", line)[0])
            learning_rate.append(lr)
    print(learning_rate)
    plt.plot(learning_rate)
    plt.show()