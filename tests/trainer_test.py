import unittest
from os.path import join

from predict360user import Dataset, Trainer, config
from predict360user.trainer import filter_df_by_entropy


class TrainerTestCase(unittest.TestCase):

  def test_init(self) -> None:
    trn = Trainer()
    self.assertEqual(trn.model_fullname, 'pos_only')
    self.assertEqual(trn.model_dir, join(config.DEFAULT_SAVEDIR, trn.model_fullname))
    self.assertEqual(trn.using_auto, False)
    trn = Trainer(dataset_name='david')
    self.assertEqual(trn.model_fullname, 'pos_only,david,,')
    self.assertEqual(trn.using_auto, False)
    for train_entropy in config.ARGS_ENTROPY_NAMES[1:] + config.ARGS_ENTROPY_AUTO_NAMES:
      trn = Trainer(train_entropy=train_entropy)
      entropy_type = 'hmpS' if train_entropy.endswith('hmp') else 'actS'
      train_entropy = train_entropy.removesuffix('_hmp')
      model_fullname = f'pos_only,all,{entropy_type},{train_entropy}'
      self.assertEqual(trn.model_fullname, model_fullname)
      self.assertEqual(trn.model_dir, join(config.DEFAULT_SAVEDIR, model_fullname))
      self.assertEqual(trn.using_auto, train_entropy.startswith('auto'))

  def test_filter_df_by_entropy(self) -> None:
    ds = Dataset()
    self.assertFalse(ds.df.empty)
    min_size = ds.df['actS_c'].value_counts().min()
    for train_entropy in ['allminsize', 'nohigh', 'nolow', 'medium', 'low', 'high']:
      fdf = filter_df_by_entropy(df=ds.df, entropy_type='actS', train_entropy=train_entropy)
      self.assertAlmostEqual(min_size, len(fdf), delta=2)

  def test_partition(self) -> None:
    trn = Trainer(train_entropy='all')
    # all
    trn.train_entropy = 'all'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    # low
    trn.train_entropy = 'low'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    # medium
    trn.train_entropy = 'medium'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    # nolow
    trn.train_entropy = 'nolow'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium','high']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium','high']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    # nohigh
    trn.train_entropy = 'nohigh'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
    # high
    trn.train_entropy = 'high'
    trn._partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['high']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['high']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'high']))
