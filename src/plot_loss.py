import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

file_name = "model/loss_log/ImitationModel_5_loss.csv"
Mix_file_name = "model/loss_log/ImitationModel_Mix_5_loss.csv"
Def_file_name = "model/loss_log/ImitationModel_Def30_loss.csv"
Norm_file_name = "model/loss_log/ImitationModel_Norm30_loss.csv"
Agg_file_name = "model/loss_log/ImitationModel_Agg30_loss.csv"
A_name = "model/loss_log/ImitationModel_Agg_e100_b128_deep_loss.csv"
Aggv2_deep_name = "model/loss_log/ImitationModel_Agg_e100_b128_deep_range55_ExponentialLR_loss.csv"
Agg_e200_cos = "model/loss_log/ImitationModel_Agg_e200_cos_2_loss.csv"
Agg_e300 = "model/loss_log/ImitationModel_Agg_e300_ExponentialLR2_loss.csv"
Agg_e300_b8192 = "model/loss_log/ImitationModel_Agg_e300_batch8192_ExponentialLR2_loss.csv"

data = pd.read_csv(file_name)
data = np.array(data)
mix = np.array(pd.read_csv(Mix_file_name))
def30 = np.array(pd.read_csv(Def_file_name))
norm30 = np.array(pd.read_csv(Norm_file_name))
agg30 = np.array(pd.read_csv(Agg_file_name))
A = np.array(pd.read_csv(A_name))
Mix_deep_name = np.array(pd.read_csv(Aggv2_deep_name))
Agg_e200_cos = np.array(pd.read_csv(Agg_e200_cos))
Agg_e300 = np.array(pd.read_csv(Agg_e300))
Agg_e300_b8192 = np.array(pd.read_csv(Agg_e300_b8192))
# 平滑
def smooth(data, weight=0.9):
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
plt.plot(A[:,0],smooth(A[:,1]))
plt.plot(Mix_deep_name[:,0],smooth(Mix_deep_name[:,1]))
plt.plot(Agg_e200_cos[:,0],smooth(Agg_e200_cos[:,1]))
plt.legend(["ImitationModel_Mix_5","ImitationModel_Def30","ImitationModel_Norm30","ImitationModel_Agg30","ImitationModel_Agg_e100_b128_deep","ImitationModel_Agg_e100_b128_deep_range55_ExponentialLR_loss"])


plt.title("smoothed loss")
plt.loglog()

plt.figure(1)
plt.plot(smooth(Agg_e200_cos[:,0]))
plt.plot(smooth(Agg_e300[:,0]))
plt.plot(smooth(Agg_e300_b8192[:,0]))
plt.legend(["ImitationModel_Agg_e200_cos","ImitationModel_Agg_e300","ImitationModel_Agg_e300_b8192"])
plt.loglog()

plt.show()