from predict360user.train import Config
from predict360user.start_train import train_and_eval
import logging


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )
    cfg = Config()
    cfg.model_name = "TRACK"
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
