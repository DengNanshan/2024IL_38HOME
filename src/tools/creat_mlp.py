import torch.nn as nn

def creat_mlp(input,
              layers,
              output,
              activation_fn = nn.ReLU):
    if len(layers)>0:
        modules = [nn.Linear(input,layers[0]),activation_fn]
    else:
        modules=[]

    for i in range (len(layers)-1):
        modules.append(nn.Linear(layers[i],layers[i+1]))
        modules.append(activation_fn)

    modules.append(nn.Linear(layers[-1],output))

    return modules
