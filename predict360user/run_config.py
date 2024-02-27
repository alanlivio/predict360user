import random
from dataclasses import MISSING, dataclass

import numpy as np
import tensorflow as tf

import wandb


@dataclass
class RunConfig:
    """Run configuration.

    Keyword arguments:
    name: -- short descriptive name, useful for wandb grouping seeds
    batch_size  -- model batch size
    dataset  -- dataset name from .data_ingestion.DATASETS
    epochs  -- model training epochs
    initial_epoch -- training epoch, useful when resuming
    gpu_id  -- traing gpu id
    h_window  -- model prediction horizon window size
    init_window  -- init buffer window size
    lr  -- model learning rate
    m_window  -- model memory window learning rate
    model  -- model name from .models folder
    train_size  -- model training size
    test_size  -- model test size
    """

    name = None
    batch_size: int = 128
    dataset: str = "all"
    epochs: int = 30
    gpu_id: str = ""
    h_window: int = 25
    init_window: int = 30
    lr: float = 0.0005
    m_window: int = 5
    model: str = "pos_only"
    train_size: float = 0.8
    test_size: float = 0.2
    initial_epoch = 0
    seed: int = 0

def set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
