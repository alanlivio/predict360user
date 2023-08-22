from predict360user.trainer import Trainer, TrainerCfg
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="trainer")
def main(cfg: TrainerCfg) -> None:
    trn = Trainer(cfg)
    trn.run()


if __name__ == "__main__":
    main()
