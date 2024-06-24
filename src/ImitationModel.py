import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

class ImitationModel(nn.Module):
    def __init__(self):
        super(ImitationModel, self).__init__()
        configparser = self.default_config()

        self.fc1 = nn.Linear(configparser["input_size"], configparser["hidden_size"])


        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.count = 0
        self.losss = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print("ImitationModel device",self.device)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "input_size": 5,
                "hidden_size": 128,
                "output_size": 5,
            }
        )
        return config

    @classmethod
    def load_config(cls, config):
        """
        还没有用到
        :param config:
        :return:
        """
        self.fc1 = nn.Linear(config["input_size"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.fc3 = nn.Linear(config["hidden_size"], config["output_size"])
        self.count = 0
        self.losss = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print("device",self.device)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, states, actions, epochs=100, batch_size=32,lr=0.001):
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
        plt.show()

if __name__ == "__main__":
    # generate dataset
    states = torch.randn(1090, 5)
    actions = states

    # generate model
    model = ImitationModel()

    # load config
    with open("conf/ImitationModel.json", "r") as f:
        conf_str = f.read()
    conf = json.loads(conf_str)
    model.load_config(conf)

    # train model
    model.train(states, actions)

    # draw loss
    model.draw_loss()