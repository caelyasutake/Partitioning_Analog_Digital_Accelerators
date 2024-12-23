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

def create_rpu_config(g_max=25, tile_size=512, modifier_std=0.07):
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = tile_size
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)
    rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.type = WeightModifierType.MULT_NORMAL
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.modifier.std_dev = modifier_std
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = 0.05
    rpu_config.forward.inp_res = 1 / (2 ** 8 - 2)
    rpu_config.forward.out_res = 1 / (2 ** 8 - 2)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

def compute_hardware_metrics(rpu_config, outputs):
    temperature = float(np.random.uniform(0.8, 1.2) * rpu_config.forward.inp_res * rpu_config.forward.out_res)
    noise = float(np.random.uniform(0.8, 1.2) * rpu_config.forward.out_noise)
    drift = float(rpu_config.drift_compensation.readout(outputs))
    return temperature, noise, drift

def compute_hardware_loss(rpu_config, outputs):
    temperature, noise, drift = compute_hardware_metrics(rpu_config, outputs)
    temperature = torch.tensor(temperature, device=outputs.device)
    hardware_loss = 0.2 * temperature + 0.3 * noise + 0.2 * drift
    return hardware_loss

def update_rpu_config(rpu_config, perturbation_scale=0.01):
    rpu_config.forward.inp_res += rpu_config.forward.inp_res * perturbation_scale
    rpu_config.forward.out_res += rpu_config.forward.inp_res * perturbation_scale
    rpu_config.forward.out_noise += rpu_config.forward.inp_res * perturbation_scale
    g_max = rpu_config.noise_model.g_max + rpu_config.noise_model.g_max * perturbation_scale
    rpu_config.noise_model = PCMLikeNoiseModel(g_max)
    rpu_config.modifier.std_dev += rpu_config.modifier.std_dev * perturbation_scale

def create_sgd_optimizer(model, learning_rate):
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)
    return optimizer
