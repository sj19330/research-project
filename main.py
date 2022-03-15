import torch
import numpy as np
import random as r


def differenceModelDV(sensorySignal):
    return sensorySignal[1].item() - sensorySignal[0].item()


def maxModelDV(sensorySignal):
    return differenceModelDV(sensorySignal)


def getData(dataLen, noise, datacheck):
    dataset = torch.empty(0)
    labels = torch.empty(0)
    s_values = [0.015, 0.035, 0.07, 0.15]
    variance = noise ** 2 / 2
    for i in range(dataLen):
        s_index = r.randint(0, 3)
        l = r.randint(1, 2)
        s = s_values[s_index]
        targetValue = np.random.normal(s, variance)
        normalValue = np.random.normal(0, variance)
        if datacheck:
            print("ta rget    value: ", targetValue,"Normal value: ", normalValue, "     s = ", s, "      label = ", l)
        if l == 1:
            sensorySignal = torch.tensor([[targetValue, normalValue]])
        elif l == 2:
            sensorySignal = torch.tensor([[normalValue, targetValue]])
        else:
            print("ERROR: l is out of range")
        dataset = torch.cat((dataset, sensorySignal), 0)
        labels = torch.cat((labels, torch.tensor([l])), 0)
        if (i * 100 / dataLen) % 5 == 0:
            print(i * 100 / dataLen, "% done making data")
    print("sensory signals are made")
    return dataset, labels


if __name__ == '__main__':
    dataLen = 10000
    noise = 0.2
    dataset, labels = getData(dataLen, noise, False)
    print(dataset, len(dataset), labels, len(labels))
    i = 0
    for data, label in zip(dataset, labels):
        if data[0].item() < data[1].item():
            if label == 1:
                i = i + 1
        else:
            if label == 2:
                i = i + 1
    print(i / dataLen)
