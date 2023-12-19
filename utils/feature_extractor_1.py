import torch
import torch.nn as nn


class g_function(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(18,16,(1,9))
        self.max1 = nn.MaxPool2d((1,2))
        self.conv2 = nn.Conv2d(16, 32,(1,9))
        self.max2 = nn.MaxPool2d((1,3))
        self.linear1 = nn.Linear(2592, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256,64)
        self.flatten = nn.Flatten()

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.max1(x)
        x = self.relu(self.conv2(x))
        x = self.max2(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(torch.squeeze(x)))
        x = self.linear2(x)
        return torch.squeeze(x)
