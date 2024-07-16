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
Agg_v3 = "model/loss_log/ImitationModel_Agg_v3_loss.csv"
Norm_v3 = "model/loss_log/ImitationModel_Norm_v3_loss.csv"
Def_v3 = "model/loss_log/ImitationModel_Def_v3_loss.csv"
Mix_v3_033 = "model/loss_log/ImitationModel_Mix_v3_033_loss.csv"
Mix_v3 = "model/loss_log/ImitationModel_Mix_v3_full_loss.csv"

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
Agg_v3 = np.array(pd.read_csv(Agg_v3))
Norm_v3 = np.array(pd.read_csv(Norm_v3))
Def_v3 = np.array(pd.read_csv(Def_v3))
Mix_v3_033 = np.array(pd.read_csv(Mix_v3_033))
Mix_v3 = np.array(pd.read_csv(Mix_v3))

# 平滑
def smooth(data, weight=0.99):
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
plt.plot(Agg_v3[:,0],smooth(Agg_v3[:,1]))
plt.plot(Norm_v3[:,0],smooth(Norm_v3[:,1]))
plt.plot(Def_v3[:,0],smooth(Def_v3[:,1]))


plt.legend(["ImitationModel_Mix_5","ImitationModel_Def30","ImitationModel_Norm30","ImitationModel_Agg30","ImitationModel_Agg_e100_b128_deep","ImitationModel_Agg_e100_b128_deep_range55_ExponentialLR_loss",
            "ImitationModel_Agg_v3",
            "ImitationModel_Norm_v3",
            "ImitationModel_Def_v3"])


plt.title("smoothed loss")
plt.loglog()

plt.figure(1)
plt.plot(Agg_v3[:,0],smooth(Agg_v3[:,1]))
plt.plot(Norm_v3[:,0],smooth(Norm_v3[:,1]))
plt.plot(Def_v3[:,0],smooth(Def_v3[:,1]))
plt.plot(Mix_v3_033[:,0],smooth(Mix_v3_033[:,1]))
plt.plot(Mix_v3[:,0],smooth(Mix_v3[:,1]))

plt.legend(["ImitationModel_Agg_v3",
            "ImitationModel_Norm_v3",
            "ImitationModel_Def_v3",
            "ImitationModel_Mix_v3_033",
            "ImitationModel_Mix_v3"])
plt.semilogy()
# plt.loglog()

plt.show()