import numpy as np
import pandas as pd
import glob
import shutil
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


from utils.test import Tester
from utils.feature_extractor import g_function

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset, random_split
import torch
from utils.pamap_dataset import PAMAP
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from utils.classifier import Classifier
import time

def same_model(model_g, model_g_1):
    are_same = True
    for (param_1, value_1), (param_2, value_2) in zip(model_g.state_dict().items(), model_g_1.state_dict().items()):
        if param_1 != param_2 or not torch.equal(value_1, value_2):
            are_same = False
            break

    #print("Models are the same:" if are_same else "Models are different")
    return are_same

def split(dataset, samples):
    counts = {i: 0 for i in range(6)}  # Assuming labels are from 0 to 11
    training = []
    testing = []

    for a in dataset:
        label = a[1]
        print(label)
        if counts[label] < samples:
            training.append(a)
            counts[label] += 1
        else:
            testing.append(a)

    return training, testing


def load_MHEALTH(train, validation, test):
    dataset = PAMAP(train=train, validation=validation, test= test )
    dataset.get_datasets()
    dataset.preprocessing()
    dataset.normalize()
    dataset.data_segmentation()
    dataset.prepare_dataset()

    return dataset.training_final, dataset.validation_final, dataset.testing_final

def experiment(distribution):
    dist = {1:([3,4,5,6,7,8], [2], [1]),
            2:([3,4,5,6,7,8], [1], [2]),
            3:([1,4,5,6,7,8], [2], [3]),
            4:([1,3,5,6,7,8], [2], [4]),
            5:([1,3,4,6,7,8], [2], [5]),
            6:([1,3,4,5,7,8], [2], [6]),
            7:([1,3,4,5,6,8], [2], [7]),
            8:([1,3,4,5,6,7], [2], [8])
            }
    train,val,test = dist[distribution]
    return train, val, test
    
    
train,val,t = experiment(1)
training, validation, test = load_MHEALTH(train,val,t)

uci_har_training = []
uci_har_validation = []
uci_har_test = []


for a in training:
    uci_har_training.append((a[0],a[1]))
for b in validation:
    uci_har_validation.append((b[0],b[1]))
for c in test:
    uci_har_test.append((c[0],c[1]))
