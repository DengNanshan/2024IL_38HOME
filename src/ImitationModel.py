import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

class ImitationModel(nn.Module):
    def __init__(self, config= None):
        super(ImitationModel, self).__init__()
        self.conf = self.default_config()
        if config:
            self.conf = self.load_config(config)
        self.layers = nn.ModuleList()
        print(self.conf["hidden_size"])
        print(len(self.conf["hidden_size"]))
        if len(self.conf["hidden_size"]) >0:
            self.layers.append(nn.Linear(self.conf["input_size"], self.conf["hidden_size"][0]))
            for i in (1,len(self.conf["hidden_size"])-1):
                self.layers.append(nn.Linear(self.conf["hidden_size"][i-1],self.conf["hidden_size"][i]))
            self.layers.append(nn.Linear(self.conf["hidden_size"][-1],self.conf["output_size"]))
        else:
            raise ValueError("hidden_size Error")
        self.count = 0
        self.losss = []
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

    def train(self, states, actions, epochs=100, batch_size=32,lr=0.0001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(torch.Tensor(states).to(self.device), torch.Tensor(actions).to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, (state, action) in enumerate(dataloader):
                optimizer.zero_grad()
                output = self(state)
                loss = criterion(output, action)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    self.count = self.count + 100
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
                    self.losss.append([self.count,loss.item()])
    def save(self, path):
        torch.save(self.state_dict(), path)

    def draw_loss(self):
        losss = np.array(self.losss)
        plt.plot(losss[:,0], losss[:, 1])
        plt.loglog
        plt.show()

if __name__ == "__main__":
    # generate dataset
    states = torch.randn(1090, 5)
    actions = states



    # load config
    with open("conf/ImitationModel.json", "r") as f:
        conf_str = f.read()
    conf = json.loads(conf_str)

    # generate model
    model = ImitationModel(config=conf)


    # train model
    model.train(states, actions)

    # draw loss
    model.draw_loss()