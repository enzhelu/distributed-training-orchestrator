# Distributed Training Orchestrator

![DeepSpeed](https://img.shields.io/badge/Optimization-DeepSpeed-green.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Distributed](https://img.shields.io/badge/Scale-Multi--Node-blue.svg)

A framework for orchestrating large-scale distributed training of foundation models. This system simplifies the integration of **DeepSpeed** and **Megatron-LM** for training models with billions of parameters across multi-node GPU clusters.

## 🌟 Key Features

- **Configuration Management**: Centralized YAML-based configuration for ZeRO stages, pipeline parallelism, and tensor parallelism.
- **Node Monitoring**: Real-time logging of GPU utilization, memory bandwidth, and communication overhead.
- **Checkpointing Suite**: Automated sharded checkpointing and seamless recovery logic for long-running jobs.
- **Launcher Templates**: Pre-configured SLURM and Kubernetes scripts for cloud-native or on-prem clusters.

## 📂 Repository Structure

- `src/`: Core logic for model sharding and distributed initialization.
- `configs/`: DeepSpeed JSON and Training YAML files.
- `launchers/`: Scripts for `torchrun`, `deepspeed`, and cluster job managers.
- `scripts/`: Helper utilities for dataset preparation and model conversion.

## 🛠️ Requirements

- PyTorch 2.0+
- DeepSpeed >= 0.10.0
- NVIDIA NCCL

## 🚀 Usage

```bash
# Launch a multi-node training job
bash launchers/run_cluster.sh --config configs/zero3_offload.json
```

---
*Developed by Enzhe Lu | AI Researcher @ Moonshot AI*
