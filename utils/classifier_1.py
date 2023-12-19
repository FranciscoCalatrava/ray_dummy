import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)

class Classifier(nn.Module):
    def __init__(self, D_in=128):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding = 'same')
        self.maxpool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding = 'same')
        self.avgpool  = nn.AvgPool1d(5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.linear1 = nn.Linear(192, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 12)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x