import torch
import torch.nn as nn


class g_function(nn.Module):
    def __init__(self, input_shape , num_blocks = []):
        super(g_function, self).__init__()
        in_channels, height, width = input_shape

        self.in_channels = in_channels
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(18, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = (1,3), stride=(1,2), padding=(0,1), dilation=1, return_indices=False, ceil_mode=False)

        self.layer1 = self._make_layer_1D(64, num_blocks[0],  stride=1)
        self.layer2 = self._make_layer_1D(128, num_blocks[1], stride=(1,2))


        self.adpavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,64)
        self.tanh = nn.Tanh()

    def _make_layer_1D(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.ModuleList()
        layers.append(BasicBlock_1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock_1D.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.adpavgpool(x)
        x = self.flatten(x)
        x = self.tanh(x)
        # x = self.linear1(x)
        return x

class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out