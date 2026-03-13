import os
import torch
import deepspeed

def initialize_training(model, train_dataset, config_path):
    """Initializes distributed training using DeepSpeed."""
    print(f"Loading DeepSpeed configuration from {config_path}")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        config=config_path
    )
    return model_engine

if __name__ == "__main__":
    print("Distributed Training Module Initialized.")
