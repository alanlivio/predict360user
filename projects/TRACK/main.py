from predict360user.train import Trainer, TrainerCfg
import hydra


@hydra.main(
    version_base=None, config_path="../../predict360user/configs", config_name="trainer"
)
def main(cfg: TrainerCfg) -> None:
    cfg.model_name = "TRACK"
    trn = Trainer(cfg)
    trn.run()


if __name__ == "__main__":
    main()
