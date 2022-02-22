import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pyro
from pyro.distributions.util import broadcast_shape
import torch.nn as nn 

def plot_batch_with_labels(data, classes):
    figure = plt.figure(figsize=(12, 10))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        img, label = data[i]
        figure.add_subplot(rows, cols, i)
        plt.title(classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    pyro.set_rng_seed(seed)

from torch.utils.data import DataLoader
def get_batch(data, size, device):
    batch = None
    for x, y in DataLoader(data, batch_size=size, shuffle=False, num_workers=0):
        batch = (x.to(device),y.to(device))
        break
    return batch

def plot_batch(x, rows=5, cols=5, figsize=(12, 10)):
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        img = x[i-1,0,:,:]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
   
class ConcatModule(nn.Module):
    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            #print('in', [s.shape for s in input_args])
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
                #print('broadcast', [s.shape for s in input_args])
            out = torch.cat(input_args, dim=-1)
            #print('out', out.shape)
            return out