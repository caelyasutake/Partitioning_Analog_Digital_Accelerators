import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,random_split,Dataset
import torch.optim as optim
from tqdm import tqdm
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
    WeightRemapType,
)
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.parameters import IOParameters

class DynamicLayer(nn.Module):
    def __init__(self, layer_type, *args, **kwargs):

        super(DynamicLayer, self).__init__()

        self.layer_type = layer_type
        self.args = args
        self.kwargs = kwargs

        rpu_config = create_rpu_config()

        if layer_type == 'conv':
            self.analog_layer = AnalogConv2d(*args, **kwargs, rpu_config=rpu_config)
            self.digital_layer = nn.Conv2d(*args, **kwargs)
        elif layer_type == 'linear':
            self.analog_layer = AnalogLinear(*args, **kwargs, rpu_config=rpu_config)
            self.digital_layer = nn.Linear(*args, **kwargs)
        else:
            raise ValueError("Invalid layer_type. Use 'conv' or 'linear'.")

        self.is_analog = False

    def forward(self, x):
        if self.is_analog:
            return self.analog_layer(x)
        else:
            return self.digital_layer(x)

    def toggle(self, is_analog):

        self.is_analog = is_analog

        # If switching to analog, synchronize weights
        if self.is_analog:
            if self.layer_type == 'conv':
                self.analog_layer.set_weights(self.digital_layer.weight.detach().cpu().numpy().reshape(-1))
            elif self.layer_type == 'linear':
                self.analog_layer.set_weights(self.digital_layer.weight.detach().cpu().numpy(),
                                               self.digital_layer.bias.detach().cpu().numpy())

        # If switching to digital, synchronize weights
        else:
            if self.layer_type == 'conv':
                weights = torch.tensor(self.analog_layer.get_weights()[0]).reshape(self.digital_layer.weight.shape)
                self.digital_layer.weight.data.copy_(weights)
            elif self.layer_type == 'linear':
                weights, bias = self.analog_layer.get_weights()
                self.digital_layer.weight.data.copy_(torch.tensor(weights))
                self.digital_layer.bias.data.copy_(torch.tensor(bias))

class MTL_Net_DynamicToggle(nn.Module):
    def __init__(self, input_channel, num_class):
        super(MTL_Net_DynamicToggle, self).__init__()
        self.classes = num_class

        # Replace layers with dynamic layers
        self.conv1 = DynamicLayer('conv', in_channels=input_channel, out_channels=8, kernel_size=3, stride=1)
        self.conv2 = DynamicLayer('conv', in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = DynamicLayer('linear', 64, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = DynamicLayer('linear', 256, 128)
        self.dropout2 = nn.Dropout(0.3)

        # Task-specific layers remain digital
        self.fc3 = nn.Linear(128, self.classes[0])
        self.fc4 = nn.Linear(128, self.classes[1])

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3)
        x = F.relu(self.fc1(x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x1 = self.fc3(x)  # Task 1 output
        x2 = self.fc4(x)  # Task 2 output

        return x1, x2

    def toggle_layer(self, layer_name, is_analog):
        """
        Toggle a specific layer between analog and digital.
        """
        getattr(self, layer_name).toggle(is_analog)