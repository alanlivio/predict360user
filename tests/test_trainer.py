import unittest
from os.path import join
from hydra import initialize, compose
from predict360user.trainer import *
from predict360user.utils import DEFAULT_SAVEDIR


class TrainerTestCase(unittest.TestCase):
    def test_init(self) -> None:
        trn = Trainer(TrainerCfg())
        self.assertEqual(trn.model_fullname, "pos_only")
        self.assertEqual(trn.model_dir, join(DEFAULT_SAVEDIR, trn.model_fullname))
        self.assertEqual(trn.using_auto, False)
        trn = Trainer(TrainerCfg(dataset_name="david"))
        self.assertEqual(trn.model_fullname, "pos_only,david,,")
        self.assertEqual(trn.using_auto, False)
        for train_entropy in ARGS_ENTROPY_NAMES[1:] + ARGS_ENTROPY_AUTO_NAMES:
            trn = Trainer(TrainerCfg(train_entropy=train_entropy))
            model_fullname = f"pos_only,all,actS,{train_entropy}"
            self.assertEqual(trn.model_fullname, model_fullname)
            self.assertEqual(trn.model_dir, join(DEFAULT_SAVEDIR, model_fullname))
            self.assertEqual(trn.using_auto, train_entropy.startswith("auto"))

    def test_init_cli(self) -> None:
        with initialize(version_base=None, config_path="../predict360user/configs"):
            # load from yaml
            cfg = compose(config_name="trainer")
            assert cfg
            # check has default values
            trn_cfg = TrainerCfg(**cfg)
            trn_cfg_default = TrainerCfg()
            self.assertEqual(trn_cfg, trn_cfg_default)
            # check some values
            trn = Trainer(cfg)
            self.assertEqual(trn.model_fullname, "pos_only")
            self.assertEqual(trn.model_dir, join(DEFAULT_SAVEDIR, trn.model_fullname))
            self.assertEqual(trn.using_auto, False)
