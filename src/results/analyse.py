import json
import matplotlib.pyplot as plt
#读取结果文件,读取所有行，第二列为agent名称，第三列为运行时间，第四列为是否结束，第五列为是Truck
import re
def get_num(x):
    return int(re.findall(r"\d+", x)[-1])

# 读取多个文件
files = {"Test_collisions_agg_v3.csv"}
data = []
times = []
names = []
for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(",")
            agent = strs[1]
            time = float(strs[2])
            names.append(agent)
            times.append(time)

# 统计每一个agent时间大于40的次数
result = []
for name in set(names):
    d = 0 # 未碰撞
    c = 0 # 碰撞
    avg_time = 0
    for i in range(len(names)):
        if names[i] == name:
            avg_time += times[i]
            if times[i] > 40:
                d += 1
            else:
                c += 1
    avg_time = avg_time/(d+c)


    print(name, "未碰撞次数", d, "碰撞次数", c)
    print("通过率", d/(d+c),"总数", d+c)
    result.append([name, d, c, d/(d+c),avg_time])

# 画图
result = sorted(result, key=lambda x: get_num(x[0]))
plt.figure(1)
plt.plot([x[0] for x in result], [x[3] for x in result])
plt.xticks(rotation=90)
plt.figure(2)
# 绘制柱状图，同一个柱上显示总数和未碰撞次数
plt.bar([x[0] for x in result], [x[2]+x[1] for x in result], label="Totle")
plt.bar([x[0] for x in result], [x[1] for x in result], label="No collision")
plt.legend()

# 绘制平均时间
plt.figure(3)
plt.plot([x[0] for x in result], [x[4] for x in result])


plt.xticks(rotation=90)
plt.show()

# 获取一个字符串中的数字


