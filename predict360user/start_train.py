from omegaconf import OmegaConf
import logging

from predict360user.model_config import Config
from predict360user.train import train_and_eval

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )
    cfg = OmegaConf.from_cli()
    train_and_eval(Config(**cfg))
