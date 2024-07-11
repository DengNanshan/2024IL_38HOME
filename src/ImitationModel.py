import glob
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR


class ImitationModel(nn.Module):
    def __init__(self, config= None):
        super(ImitationModel, self).__init__()
        self.conf = self.default_config()
        if config:
            self.conf = self.load_config(config)
        self.layers = nn.ModuleList()
        if len(self.conf["hidden_size"]) >0:
            self.layers.append(nn.Linear(self.conf["input_size"], self.conf["hidden_size"][0]))
            for i in (1,len(self.conf["hidden_size"])-1):
                self.layers.append(nn.Linear(self.conf["hidden_size"][i-1],self.conf["hidden_size"][i]))
            self.layers.append(nn.Linear(self.conf["hidden_size"][-1],self.conf["output_size"]))
        else:
            raise ValueError("hidden_size Error")
        self.count = 0
        self.losss = []
        self.learing_rate=[]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print("ImitationModel device",self.device)


    def default_config(cls) -> dict:
        config=(
            {
                "input_size": 5,
                "hidden_size": [128,128],
                "output_size": 5,
            }
        )
        return config

    # @classmethod
    def load_config(self, config=None):
        """
        还没有用到
        :param config:
        :return:
        """
        self.conf.update(config)
        print("update config",self.conf)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print("device",self.device)
        return self.conf
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def train(self, states, actions, epochs=100, batch_size=32,lr=0.0001,ir_Scheduler=False,check_point_path=None,check_rate = 10,
              loss_show = False):


        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if ir_Scheduler:
            # IL_Scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00000001)
            if ir_Scheduler == "CosineAnnealingLR":
                print("CosineAnnealingLR")
                IL_Scheduler = CosineAnnealingLR(optimizer, T_max=int(epochs/4), eta_min=0.00000001)
            else:
                IL_Scheduler = ExponentialLR(optimizer, gamma=0.9)
        dataset = torch.utils.data.TensorDataset(torch.Tensor(states).to(self.device), torch.Tensor(actions).to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        # 检查点文件路径
        checkpoint_files = None
        if check_point_path:
            checkpoint_dir = check_point_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            # 检查最后一个检查点
            checkpoint_files =glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            check_point = torch.load(latest_checkpoint)
            self.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            if ir_Scheduler:
                IL_Scheduler.load_state_dict(check_point['scheduler_state_dict'])
            start_epochs= check_point['epoch']+1
            print(f"load check_point from {latest_checkpoint}, start_epochs={start_epochs}")
        else:
            start_epochs = 0


        for epoch in range(start_epochs, epochs):
            for i, (state, action) in enumerate(dataloader):
                optimizer.zero_grad()
                output = self(state)
                loss = criterion(output, action)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    self.count = self.count + 100
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
                    if ir_Scheduler:
                        self.losss.append([self.count, loss.item(), IL_Scheduler.get_last_lr()[0]])
                    else:
                        self.losss.append([self.count, loss.item()])
            if ir_Scheduler:
                IL_Scheduler.step()
                print(f"Epoch {epoch}, Learning Rate: {IL_Scheduler.get_last_lr()[0]}")

            if (epoch % check_rate) and check_point_path == 0:
                # 保存检查点
                checkpoint_path = os.path.join(checkpoint_dir, f"ImitationModel_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': IL_Scheduler.state_dict()
                }, checkpoint_path)
                print(f"save check_point to {checkpoint_path}")
            if loss_show:
                plt.figure(1)
                plt.clf()
                losss = np.array(self.losss)
                plt.plot(losss[:, 0], losss[:, 1])
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.loglog()
                plt.show(block=False)
                plt.pause(0.4)
                plt.close()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def draw_loss(self):
        losss = np.array(self.losss)
        plt.plot(losss[:,0], losss[:, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.loglog()
        plt.show()

import random
random.seed(0)
if __name__ == "__main__":
    # generate dataset
    states = torch.randn(10900, 5)
    actions = states



    # load config
    with open("conf/ImitationModel_test.json", "r") as f:
        conf_str = f.read()
    conf = json.loads(conf_str)

    # generate model
    model = ImitationModel(config=conf)


    # train model
    model.train(states, actions,batch_size=64,epochs=1000,ir_Scheduler=True)

    # draw loss
    model.draw_loss()

""""
batch_size = 64   Epoch 99, Loss: 4.456936949281953e-05
batch_size = 328  Epoch 99, Loss: 0.0002875070204026997

"""