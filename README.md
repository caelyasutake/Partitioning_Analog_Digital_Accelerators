# Partitioning_Analog_Digital_Accelerators
This repository accompanies the research paper "Partitioning Strategies for Multi-Task Learning on Analog-Digital Heterogeneous Accelerators" by Cael Yasutake and Seojin Yoon. The project introduces a novel dynamic partitioning algorithm for Multi-Task Learning (MTL) models on analog-digital heterogeneous systems. They algorithm prioritizes energy efficiency, maintains model accuracy, and compensates for hardware-induced fluctuations such as drift and variance in analog components.

## Key Features
### Dynamic Partitioning Algorithm
We allocate MTL model layers to analog or digital components dynamically based on energy efficiency, noise levels, and computational importance. This process can be enabled for both training and inference depending on use-cases.

### Analog-Digital Architecture
This uses simulations of Analog In-Memory Computing (AIMC) for energy-efficient operations and Digital Processing Units (DPU) for stable computations. Simulations are done utilizing IBM's AIHWKit, providing experimentation on noise, drift, and physical fluctutations.

## Methodology
### Dynamic Partitioning Architecture
We implement a policy function to determine distribution of layers dynamically between analog and digital components. This policy can be tuned depending on the MTL model implementation. Further, we create a wrapper class that enables toggling of MTL layers between analog and digital nodes during runtime (training or inference).

### Datasets
Our partitioning algorithm is currently implemented on the CIFAR10 dataset for two-class classification. We are currently working on benchmarking against further, more in-depth datasets such as COCO or BelgiumTS.

### Hardware
All computations were done utilizing Google Colab Notebooks on T4 GPUs and CPUs. The analog simulations leverage PCM-based AIMC cores.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/caelyasutake/Partitioning_Analog_Digital_Accelerators
```

Currently all experiments are done in the Python notebook. This includes all necessary library installations including instructions for downloading AIHWKit. For further setup on this, please follow the [AIHWKit Installation Guide](https://aihwkit.readthedocs.io/en/latest/install.html) for further reference on this process on analog hardware simulation.

## Future Work
We would like to continue further refine the policy function through the use of more complex balancing systems, hoping to enable better performance over long-term simulations. We also plan on extending the framework to be used to handle dynamic workloads and processes, such as real-time streaming of data. Finally, we would like to benchmark more extensively on more complex datasets to fully utilize the capabilities of analog computing hardware configurations.

## Contributors
* Cael Yasutake (Columbia University) - cty2113@columbia.edu
* Seojin Yoon (Columbia University) - sy3028@columbia.edu
