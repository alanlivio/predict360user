from predict360user.train import Config, train_and_eval


def main() -> None:
    cfg = Config()
    cfg.model_name = "TRACK"
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
