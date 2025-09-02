import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)


def clone_module(module):
    # ** Description
    #
    # Create a copy of module, whose parameters, submodules are
    # created using PyTorch's torch.clone
    #
    # TODO: decide use shallow copy or deep copy!
    # **

    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._modules = clone._modules.copy()

    if hasattr(clone, '_parameters'):
        for key in module._parameters:
            if module._parameters[key] is not None:
                param = module._parameters[key]
                cloned = param.clone()
                clone._parameters[key] = cloned

    if hasattr(clone, '_modules'):
        for key in clone._modules:
            clone._modules[key] = clone_module(
                module._modules[key]
            )
    return clone



def update_module(module, updates = None):
    #"""
    #** Description **
    #Update the paramaters of a module using GD.
    #
    #"""

    if not updates == None:          ## in this case, we won't meet this case
        params = list(module.parameters())
        if len(updates) != len(list(params)):
           warn = 'WARNING:update_module(): Paramaters and updates should have same length, but we get {} & {}'.format(len(params), len(updates))
           print(warn)
        for p, g in zip(params, updates):
            p.update = g

    #Update the params
    for key in module._parameters:
        value = module._parameters[key]
        if value is not None and hasattr(value, 'update') and value.update is not None:
            module._parameters[key] = value + value.update

    #recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key])

    return module

def calc_entropy(data, model, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    value = 0
    for task in data:
        task = np.expand_dims(task, 1)
        task = np.expand_dims(task, 1)
        tmp_value = 0
        for sample in task[ : 3]:
            pred_prob = F.softmax(model(torch.tensor(sample, dtype=torch.float, device=device)), dim = 1)
            for i in range(pred_prob.dim()):
                tmp_value += pred_prob[0,i] * torch.log2(pred_prob[0,i]) / 2  #n_classes = 2
        value += tmp_value / 3
    return -value / len(data)


