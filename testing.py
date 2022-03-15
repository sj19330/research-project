import torch
import numpy as np
import random as r

if __name__ == '__main__':
    t = torch.empty(10,2)
    for i in range(len(t)):
        t[i] = torch.tensor([ i/2, ])
    print(t)
