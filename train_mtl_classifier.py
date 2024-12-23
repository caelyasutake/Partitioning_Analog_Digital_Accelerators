from train import *
from dynamic_layers import *
from utils import *
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

def main():
    model = MTL_Net_DynamicToggle(input_channel=3, num_class=[10, 2]).to('cuda')
    
    # Start with all analog
    model.toggle_layer('conv1', is_analog=True)
    model.toggle_layer('conv2', is_analog=True)
    model.toggle_layer('fc1', is_analog=True)
    model.toggle_layer('fc2', is_analog=True)
    model.toggle_layer('fc3', is_analog=True)
    model.toggle_layer('fc4', is_analog=True)
    
    optimizer = create_sgd_optimizer(model, 0.01)
    criterion = nn.CrossEntropyLoss()
    rpu_config = create_rpu_config()
    
    order = ['conv1', 'conv1',('fc1', 'fc2'),('fc3','fc4')]
    
    for epoch in range(50):
        toggle_config = dict()
        # Dynamically toggle during training
        train_loss, l1, l2 = train_model_dynamic(model, train_loader, optimizer, criterion, epoch, 'cuda', rpu_config, toggle_config)
        if train_loss > 0.8 and order:
          val = order.pop()
          toggle_config = dict()
          if type(val) is tuple:
              if val[0] > val[1]:
                toggle_config[val[1]] = False
                order.append(val[0])
              else:
                toggle_config[val[0]] = False
                order.append(val[1])
          else:
              toggle_config[val] = False
        else:
          toggle_config = None
    
        # Dynamically toggle during testing
        test_loss, acc1, acc2 = test_model_dynamic(model, test_loader, criterion, epoch, 'cuda', toggle_config)

if __name__ == '__main__':
    main()
