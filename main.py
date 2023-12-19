##Import of the basic stuff (numpy, pandas, torch, and ray)
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray import train as supertrain
import ray

##Import of the models
from utils.feature_extractor import g_function
from utils.classifier import Classifier

##Import of the datasets
from utils.pamap_dataset import PAMAP

##Import training and testing
from utils.test import Tester
from utils.training import Trainer as TrainerStep1



ray.init()

participant = 2

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


def trani_pamap(config):
    train,val,t = experiment(participant)
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
    


    train_MNIST_dataloader = torch.utils.data.DataLoader(uci_har_training, batch_size=64, shuffle=True, num_workers = 5)
    test_USPS_dataloader = torch.utils.data.DataLoader(uci_har_validation, batch_size =  65, shuffle = True,num_workers = 5)
    test_final = torch.utils.data.DataLoader(uci_har_test, batch_size =65, shuffle = True, num_workers = 5)

    device = 'cuda:0'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    model_g_1 = g_function((64,1,128),[2,2,0,0]).to(device)
    #model_g_1 = g_function().to(device)
    classifier_1 = Classifier().to(device)

    criterion_1 = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(list(model_g_1.parameters())+list(classifier_1.parameters()), lr= config["lr"], weight_decay=1e-5)
    trainerA = TrainerStep1(model_g_1,classifier_1, train_MNIST_dataloader,test_USPS_dataloader, device, optimizer_1, criterion_1, 10, writer)
    testerA = Tester(model_g_1, classifier_1, device= device)
    for i in range(20):
        a,b= trainerA.train()
        testerA.encoder = torch.load(a)
        testerA.classifier = torch.load(b)
        acc_test, f1_m, f1_w = testerA.test(test_final)

        # Send the current training result back to Tune
        supertrain.report({"mean_accuracy": acc_test.item()})



search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
}

tuner = tune.Tuner(
    tune.with_resources(trani_pamap, {'cpu':15, 'gpu': 1}),
    param_space=search_space,
)
results = tuner.fit()

print(results.get_dataframe(filter_metric='mean_accuracy', filter_mode='max'))
