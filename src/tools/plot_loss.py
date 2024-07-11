import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

file_name = "model/ImitationModel_5_loss.csv"
Mix_file_name = "model/ImitationModel_Mix_5_loss.csv"
Def_file_name = "model/ImitationModel_Def30_loss.csv"
Norm_file_name = "model/ImitationModel_Norm30_loss.csv"
Agg_file_name = "model/ImitationModel_Agg30_loss.csv"



data = pd.read_csv(file_name)
data = np.array(data)
mix = np.array(pd.read_csv(Mix_file_name))
def30 = np.array(pd.read_csv(Def_file_name))
norm30 = np.array(pd.read_csv(Norm_file_name))
agg30 = np.array(pd.read_csv(Agg_file_name))

# 平滑
def smooth(data, weight=0):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


plt.figure(2)
# plt.plot(data[:,0],smooth(data[:,1]))
plt.plot(mix[:,0],smooth(mix[:,1]))
plt.plot(def30[:,0],smooth(def30[:,1]))
plt.plot(norm30[:,0],smooth(norm30[:,1]))
plt.plot(agg30[:,0],smooth(agg30[:,1]))
plt.legend(["ImitationModel_Mix_5","ImitationModel_Def30","ImitationModel_Norm30","ImitationModel_Agg30"])



plt.title("smoothed loss")
plt.loglog()
plt.show()