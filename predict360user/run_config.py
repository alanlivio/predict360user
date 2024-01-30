from dataclasses import dataclass

@dataclass
class RunConfig:
    """Model config.
    
    Keyword arguments:
    batch_size  -- model batch size
    dataset_name  -- dataset name from .data_ingestion.DATASETS
    epochs  -- model training epochs
    gpu_id  -- traing gpu id
    h_window  -- model prediction horizon window size
    init_window  -- init buffer window size
    lr  -- model learning rate
    m_window  -- model memory window learning rate
    model_name  -- model name
    savedir  -- model directory for save file
    train_size  -- model training size
    test_size  -- model test size
    """

    batch_size: int = 128
    dataset_name: str = "all"
    epochs: int = 30
    gpu_id: str = ""
    h_window: int = 25
    init_window: int = 30
    lr: float = 0.0005
    m_window: int = 5
    model_name: str = "pos_only"
    savedir: str = "saved"
    train_size: float = 0.8
    test_size: float = 0.2
