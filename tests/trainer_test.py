import unittest
from os.path import join

from users360 import config
from users360.dataset import Dataset
from users360.trainer import (Trainer, count_traject_entropy_classes,
                              get_train_test_split)
from users360.utils.fov import calc_actual_entropy


class Test(unittest.TestCase):

  def test_init(self) -> None:
    trn = Trainer()
    self.assertEqual(trn.model_fullname, 'pos_only')
    self.assertEqual(trn.model_dir, join(config.DATADIR, trn.model_fullname))
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
      self.assertEqual(trn.model_dir, join(config.DATADIR, model_fullname))
      self.assertEqual(trn.using_auto, train_entropy.startswith('auto'))

  def test_train_test_split(self) -> None:
    self.ds = Dataset()
    self.assertTrue(self.ds.df.size)

    if not 'actS_c' in self.ds.df.columns:
      calc_actual_entropy(self.ds.df)

    # x_train all
    x_train, x_test = get_train_test_split(self.ds.df, 'all', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['low', 'medium', 'hight']))
    unique_s = set(x_test['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['low', 'medium', 'hight']))

    # x_train low
    x_train, x_test = get_train_test_split(self.ds.df, 'low', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['low']))
    unique_s = set(x_test['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['low']))

    # x_train medium
    x_train, x_test = get_train_test_split(self.ds.df, 'medium', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['medium']))
    unique_s = set(x_test['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['medium']))

    # x_train hight
    x_train, x_test = get_train_test_split(self.ds.df, 'hight', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['hight']))
    unique_s = set(x_test['actS_c'].unique())
    self.assertSequenceEqual(unique_s, set(['hight']))

  def test_partition(self) -> None:
    trn = Trainer()
    trn.partition()
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_train), (10301, 6125, 3133, 1043))
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_test), (2576, 1510, 814, 252))
    trn = Trainer(train_entropy='low')
    trn.partition()
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_train), (6108, 6108, 0, 0))
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_test), (1527, 1527, 0, 0))
    trn = Trainer(train_entropy='medium')
    trn.partition()
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_train), (3157, 0, 3157, 0))
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_test), (790, 0, 790, 0))
    trn = Trainer(train_entropy='hight')
    trn.partition()
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_train), (1036, 0, 0, 1036))
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_test), (259, 0, 0, 259))
    trn = Trainer(train_entropy='nohight')
    trn.partition()
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_train), (9265, 6089, 3176, 0))
    self.assertSequenceEqual(count_traject_entropy_classes(trn.x_test), (2317, 1546, 771, 0))