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

  def test_filter_df_by_entropy(self) -> None:
    ds = Dataset()
    self.assertFalse(ds.df.empty)
    min_size = ds.df['actS_c'].value_counts().min()
    for train_entropy in ARGS_ENTROPY_NAMES[1:]:
      fdf = filter_df_by_entropy(df=ds.df, entropy_type='actS', train_entropy=train_entropy)
      self.assertAlmostEqual(min_size, len(fdf), delta=2)

  def test_partition(self) -> None:
    with initialize(version_base=None, config_path="conf"):
      cfg = compose( config_name="config")
      exp = Experiment(cfg)
      # all
      exp.train_entropy = 'all'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      # low
      exp.train_entropy = 'low'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low']))
      classes = set(exp.x_val['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      # medium
      exp.train_entropy = 'medium'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['medium']))
      classes = set(exp.x_val['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['medium']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      # nolow
      exp.train_entropy = 'nolow'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['medium','high']))
      classes = set(exp.x_val['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['medium','high']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      # nohigh
      exp.train_entropy = 'nohigh'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low','medium']))
      classes = set(exp.x_val['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low','medium']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
      # high
      exp.train_entropy = 'high'
      exp._partition()
      self.assertGreater(len(exp.x_train), len(exp.x_val))
      classes = set(exp.x_train['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['high']))
      classes = set(exp.x_val['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['high']))
      classes = set(exp.x_test['actS_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
