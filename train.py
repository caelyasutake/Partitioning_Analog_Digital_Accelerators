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

def train_model_dynamic(model, trainloader, optim, criterion, epoch, device, rpu_config, toggle_config=None):
    """
    Train function with dynamic analog/digital toggling.

    Parameters:
    - toggle_config: dict, specifies which layers to toggle and when during training.
                     Example: {'conv1': True, 'fc1': False}.
    """
    model.train()  # Ensure the model is in training mode
    train_loss, total, total_correct1, total_correct2 = 0, 0, 0, 0

    # Apply toggling at the start of training (if toggle_config is provided)
    if toggle_config:
        for layer_name, is_analog in toggle_config.items():
            model.toggle_layer(layer_name, is_analog)

    for i, (inputs, tg1, tg2) in enumerate(tqdm(trainloader)):
        inputs, tg1, tg2 = inputs.to(device), tg1.to(device), tg2.to(device)
        optim.zero_grad()

        # Forward pass through the model
        op1, op2 = model(inputs)

        # Compute losses for both tasks
        loss1 = criterion(op1, tg1)
        loss2 = criterion(op2, tg2)
        total_loss = loss1 + loss2

        # Backpropagation
        total_loss.backward()

        # Update weights
        optim.step()

        hardware_loss = compute_hardware_loss(model, op1) + compute_hardware_loss(model, op2)
        total_loss += hardware_loss
        
        # Accumulate metrics
        train_loss += total_loss.item()
        _, pd1 = torch.max(op1.data, 1)
        _, pd2 = torch.max(op2.data, 1)

        total_correct1 += (pd1 == tg1).sum().item()
        total_correct2 += (pd2 == tg2).sum().item()
        total += tg1.size(0)

        if total_loss < prev_loss:
            update_rpu_config(rpu_config, total_loss*0.000001)
        prev_loss = total_loss

    print("Epoch: [{}]  loss: [{:.2f}] Original_task_acc [{:.2f}] animal_vs_non_animal_acc [{:.2f}]".format(
        epoch + 1, train_loss / (i + 1),
        (total_correct1 * 100 / total),
        (total_correct2 * 100 / total)
    ))

    return train_loss / (i + 1), (total_correct1 * 100 / total), (total_correct2 * 100 / total)

def test_model_dynamic(model, testloader, criterion, epoch, device, toggle_config=None):
    """
    Test function with dynamic analog/digital toggling.

    Parameters:
    - toggle_config: dict, specifies which layers to toggle during testing.
                     Example: {'conv1': True, 'fc1': False}.
    """
    model.eval()  # Ensure the model is in evaluation mode
    test_loss, total, total_correct1, total_correct2 = 0, 0, 0, 0

    # Apply toggling at the start of testing (if toggle_config is provided)
    if toggle_config:
        for layer_name, is_analog in toggle_config.items():
            model.toggle_layer(layer_name, is_analog)

    with torch.no_grad():
        for i, (inputs, tg1, tg2) in enumerate(tqdm(testloader)):
            inputs, tg1, tg2 = inputs.to(device), tg1.to(device), tg2.to(device)

            # Forward pass through the model
            op1, op2 = model(inputs)

            # Compute losses for both tasks
            loss1 = criterion(op1, tg1)
            loss2 = criterion(op2, tg2)

            # Accumulate metrics
            test_loss += loss1.item() + loss2.item()
            _, pd1 = torch.max(op1.data, 1)
            _, pd2 = torch.max(op2.data, 1)

            total_correct1 += (pd1 == tg1).sum().item()
            total_correct2 += (pd2 == tg2).sum().item()
            total += tg1.size(0)

    # Compute accuracies
    acc1 = 100. * total_correct1 / total
    acc2 = 100. * total_correct2 / total

    # Log metrics
    print("Test Epoch: [{}]  loss: [{:.2f}] Original_task_Acc [{:.2f}] animal_vs_non_animal_acc [{:.2f}]".format(
        epoch + 1, test_loss / (i + 1), acc1, acc2
    ))

    return test_loss / (i + 1), acc1, acc2
