import hydra
from omegaconf import DictConfig

from predict360user.model_config import Config
from predict360user.model_train import train_and_eval


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_and_eval(Config(**cfg))

if __name__ == "__main__":
    main()
