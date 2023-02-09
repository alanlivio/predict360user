import unittest
from os.path import join

from users360 import config
from users360.dataset import Dataset
from users360.trainer import Trainer


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

  def test_split(self) -> None:
    # actS
    ds = Dataset()
    x_train, x_test = train_test_split_entropy(ds.df,'actS','all',0.2)
    self.assertSequenceEqual(count_entropy(x_train, 'actS'), (10301, 6125, 3133, 1043))
    self.assertSequenceEqual(count_entropy(x_test, 'actS'), (2576, 1510, 814, 252))
    x_train, x_test = train_test_split_entropy(ds.df, 'actS','low',0.2)
    self.assertSequenceEqual(count_entropy(x_train, 'actS'), (6108, 6108, 0, 0))
    self.assertSequenceEqual(count_entropy(x_test, 'actS'), (1527, 1527, 0, 0))
    x_train, x_test = train_test_split_entropy(ds.df, 'actS','medium',0.2)
    self.assertSequenceEqual(count_entropy(x_train, 'actS'), (3157, 0, 3157, 0))
    self.assertSequenceEqual(count_entropy(x_test, 'actS'), (790, 0, 790, 0))
    x_train, x_test = train_test_split_entropy(ds.df, 'actS','hight',0.2)
    self.assertSequenceEqual(count_entropy(x_train, 'actS'), (1036, 0, 0, 1036))
    self.assertSequenceEqual(count_entropy(x_test, 'actS'), (259, 0, 0, 259))
    x_train, x_test = train_test_split_entropy(ds.df, 'actS','nohight',0.2)
    self.assertSequenceEqual(count_entropy(x_train, 'actS'), (9265, 6089, 3176, 0))
    self.assertSequenceEqual(count_entropy(x_test, 'actS'), (2317, 1546, 771, 0))
    # hmpS
    # TODO: add hmpS cases