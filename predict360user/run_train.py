from predict360user.trainer import Trainer, TrainerCfg
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="trainer")
def trainer_run(cfg: TrainerCfg) -> None:
    exp = Trainer(cfg)
    exp.run()

if __name__ == "__main__":
    trainer_run()
