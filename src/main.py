import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import torch

env = gym.make('highway-v0', render_mode="rgb_array")


"""读取配置文件"""
import json
with open("conf/HighwayConf_Discrete.json", "r") as f:
    conf_str = f.read()
conf = json.loads(conf_str)
env.configure(conf)

env.reset()



for i in range(1000):
    env.render()
    """随机动作"""
    # env.step(env.action_space.sample())
    """固定动作"""
    env.step(4)
    env.env.road.vehicles[1]
    for vehicle in env.env.road.vehicles[1:]:
        print(vehicle.d_get_state())
    # print(env.env.road.vehicles[1].d_get_state())
env.close()


print(1)