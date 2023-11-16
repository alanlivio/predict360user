import hydra
from omegaconf import DictConfig

from predict360user.model_config import Config
from predict360user.train import train_and_eval


@hydra.main(version_base=None, config_path=".", config_name="start_train")
def main(cfg: DictConfig) -> None:
    train_and_eval(Config(**cfg))

if __name__ == "__main__":
    main()
