import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

file_name = "../model/ImitationModel_5_loss.csv"
data = pd.read_csv(file_name)
data = np.array(data)
plt.figure(1)
plt.plot(data[:,0],data[:,1])
plt.title("raw loss")
plt.loglog()


# 平滑
def smooth(data, weight=0.95):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

smoothed_data = smooth(data[:,1])
plt.figure(2)
plt.plot(data[:,0],smoothed_data)
plt.title("smoothed loss")
plt.loglog()
plt.show()