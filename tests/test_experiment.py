import unittest
from os.path import join
from hydra import initialize, compose
from predict360user import Dataset
from predict360user.experiment import *


class ExperimentTestCase(unittest.TestCase):

  def test_init(self) -> None:
    with initialize(version_base=None, config_path="../predict360user/conf"):
      cfg = compose(config_name="config")
      exp = Experiment(cfg.experiment)
      self.assertEqual(exp.model_fullname, 'pos_only')
      self.assertEqual(exp.model_dir, join('saved', exp.model_fullname))
      self.assertEqual(exp.using_auto, False)
    with initialize(version_base=None, config_path="../predict360user/conf"):
      cfg = compose(config_name="config")
      cfg.experiment.dataset_name = 'david'
      exp = Experiment(cfg.experiment)
      self.assertEqual(exp.model_fullname, 'pos_only,david,,')
      self.assertEqual(exp.model_dir, join('saved', exp.model_fullname))
      self.assertEqual(exp.using_auto, False)
    with initialize(version_base=None, config_path="../predict360user/conf"):
      cfg = compose(config_name="config")
      for train_entropy in ARGS_ENTROPY_NAMES[1:] + ARGS_ENTROPY_AUTO_NAMES:
        cfg.experiment.train_entropy = train_entropy
        exp = Experiment(cfg.experiment)
        model_fullname = f'pos_only,all,actS,{train_entropy}'
        self.assertEqual(exp.model_fullname, model_fullname)
        self.assertEqual(exp.model_dir, join('saved', model_fullname))
        self.assertEqual(exp.using_auto, train_entropy.startswith('auto'))