import json
import matplotlib.pyplot as plt
#读取结果文件,读取所有行，第二列为agent名称，第三列为运行时间，第四列为是否结束，第五列为是Truck
import re
def get_num(x):
    return int(re.findall(r"\d+", x)[-1])

# 读取多个文件
files = {"Test_collisions_v3.csv"}
data = []
times = []
names = []
envs = []
for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(",")
            env= strs[0]
            agent = strs[1]
            time = float(strs[2])
            names.append(agent)
            times.append(time)
            envs.append(env)

# 统计每一个agent时间大于40的次数
result = []
for env in set(envs):
    for name in set(names):
        d = 0 # 未碰撞
        c = 0 # 碰撞
        avg_time = 0
        for i in range(len(names)):
            if (names[i] == name) and (envs[i] == env):
                avg_time += times[i]
                if times[i] > 40:
                    d += 1
                else:
                    c += 1
        avg_time = avg_time/(d+c)

        print(env,name, "未碰撞次数", d, "碰撞次数", c)
        print("通过率", d/(d+c),"总数", d+c)
        result.append([env,name, d, c, d/(d+c),avg_time])


# 将result 按照环境进行分类到r_env[]中
r_env = []
envs = set([x[0] for x in result])
envs = list(envs)
for env in envs:
    r_env.append([x for x in result if x[0] == env])

# 按照r_env[2]对r_env进行排序


plt.figure(1)
# 错开柱状图
import numpy as np
x = np.arange(5)
print(x)
for i in range(len(r_env)):
    # plt.figure(i)
    plt.bar(x+i*0.1, [x[4] for x in r_env[i]], label=envs[i],width=0.1)
    # plt.bar([x[1] for x in r_env[i]], [x[4] for x in r_env[i]], label=envs[i],width=0.1,)
    plt.xticks(rotation=90)
# 修改横坐标坐标轴
plt.xticks(rotation=90)
plt.xticks(x,list(set([x[1] for x in result])),rotation=90)
plt.legend()
plt.show()








# # 画图
#
# plt.figure(1)
# plt.plot([x[0] for x in result], [x[3] for x in result])
# plt.xticks(rotation=90)
# # 绘制柱状图，同一个柱上显示总数和未碰撞次数,同一个env为一簇，绘制同名称的结果
# plt.figure(2)
# plt.bar([x[1] for x in result], [x[2]+x[3] for x in result], label="Totle")
# plt.bar([x[1] for x in result], [x[2] for x in result], label="No collision")
# plt.legend()
# plt.figure(3)
#
# plt.bar([x[1] for x in result], [x[2]+x[3] for x in result], label="Totle")
# plt.bar([x[1] for x in result], [x[2] for x in result], label="No collision")
# plt.legend()
#
# # 绘制平均时间
# plt.figure(3)
# plt.plot([x[1] for x in result], [x[5] for x in result])
#
#
# plt.xticks(rotation=90)
# plt.show()
#
# # 获取一个字符串中的数字
#

