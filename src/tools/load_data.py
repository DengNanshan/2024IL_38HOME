import pandas as pd
import ast
import numpy as np


def load_data(filename):
    print("loading data")
    data = pd.read_csv(filename)
    states = np.array([ast.literal_eval(state) for state in data["state"]])
    actions = np.array([ast.literal_eval(action) for action in data["action"]])
    print("loading data finished")
    return states, actions