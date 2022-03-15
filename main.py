import torch
import numpy as np
import random as r


def differenceModelDV(sensorySignal):
    return sensorySignal[1].item() - sensorySignal[0].item()


def maxModelDV(sensorySignal):
    return differenceModelDV(sensorySignal)


def getData(dataLen, noise, datacheck):
    dataset = torch.empty(dataLen, 2)
    labels = torch.empty(dataLen)
    s_values = [0.015, 0.035, 0.07, 0.15]
    variance = noise ** 2 / 2
    for index in range(dataLen):
        s_index = r.randint(0, 3)
        label = r.randint(1, 2)
        s = s_values[s_index]
        targetValue = np.random.normal(s, variance)
        normalValue = np.random.normal(0, variance)
        if datacheck:
            print("Target value: ", targetValue, "Normal value: ", normalValue, "     s = ", s, "      label = ", label)
        if label == 1:
            dataset[index] = torch.tensor([targetValue, normalValue])
        elif label == 2:
            dataset[index] = torch.tensor([normalValue, targetValue])
        else:
            print("ERROR: label is out of range")
            break
        labels[index] = torch.tensor([label])
        if (index * 100 / dataLen) % 5 == 0:
            print(index * 100 / dataLen, "% done making sensory signals")
    print("sensory signals are made")
    return dataset, labels


if __name__ == '__main__':
    dataLen = 1000000
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
