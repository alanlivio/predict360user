import hydra
from omegaconf import DictConfig

from predict360user.model_config import Config
from predict360user.model_train import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    trn = Trainer(Config(**cfg))
    trn.run()


if __name__ == "__main__":
    main()
