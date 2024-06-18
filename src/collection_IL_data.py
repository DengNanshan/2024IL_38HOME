import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import csv
import json
env = gym.make('highway-v0', render_mode="rgb_array")


"""读取配置文件"""
import json
with open("conf/HighwayConf_Discrete.json", "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)
env.configure(conf)

env.reset()

count =0
episode = 5
with open("data/IL_data5.csv", mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["state","action"])
    for i in range(episode):
        env.reset()
        Done = False
        Track = False
        while not Done and not Track:
            # env.render()
            """随机动作"""
            # env.step(env.action_space.sample())
            """固定动作"""
            obs, reward, Done, Track, info = env.step(4)
            for vehicle in env.env.road.vehicles[1:]:
                state = vehicle.d_get_state()
                s = state["state"].flatten()
                a = [state["action"]["steering"], state["action"]["acceleration"]]
                # 写入文件
                # writer.writerow([str(s),str(a)])
                writer.writerow([','.join(map(str, s)),','.join(map(str, a))])


            print("time",env.env.time)
env.close()


print(1)