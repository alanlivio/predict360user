from dataclasses import dataclass
import numpy as np
import random
import tensorflow as tf


@dataclass
class RunConfig:
    """Run configuration.

    Keyword arguments:
    name: -- run name, which my include other params
    batch_size  -- model batch size
    dataset  -- dataset name from .data_ingestion.DATASETS
    epochs  -- model training epochs
    gpu_id  -- traing gpu id
    h_window  -- model prediction horizon window size
    init_window  -- init buffer window size
    lr  -- model learning rate
    m_window  -- model memory window learning rate
    model  -- model name from .models folder
    train_size  -- model training size
    test_size  -- model test size
    """

    name: str
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
    seed: int = 0

    def __post_init__(self) -> None:
        self.name = f"{self.model}"

    def set_random_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
