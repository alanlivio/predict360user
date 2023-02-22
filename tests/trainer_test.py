import unittest
from os.path import join

from users360 import config
from users360.dataset import Dataset
from users360.trainer import Trainer, train_test_split_entropy


class Test(unittest.TestCase):

  def test_init(self) -> None:
    trn = Trainer()
    self.assertEqual(trn.model_fullname, 'pos_only')
    self.assertEqual(trn.model_dir, join(Dataset.DATADIR, trn.model_fullname))
    self.assertEqual(trn.using_auto, False)
    trn = Trainer(dataset_name='david')
    self.assertEqual(trn.model_fullname, 'pos_only,david,,')
    self.assertEqual(trn.using_auto, False)
    for train_entropy in Trainer.ARGS_ENTROPY_NAMES[1:] + Trainer.ARGS_ENTROPY_AUTO_NAMES:
      trn = Trainer(train_entropy=train_entropy)
      entropy_type = 'hmpS' if train_entropy.endswith('hmp') else 'actS'
      train_entropy = train_entropy.removesuffix('_hmp')
      model_fullname = f'pos_only,all,{entropy_type},{train_entropy}'
      self.assertEqual(trn.model_fullname, model_fullname)
      self.assertEqual(trn.model_dir, join(Dataset.DATADIR, model_fullname))
      self.assertEqual(trn.using_auto, train_entropy.startswith('auto'))

  def test_train_test_split_entropy(self) -> None:
    self.ds = Dataset()
    self.assertTrue(self.ds.df.size)

    if not 'actS_c' in self.ds.df.columns:
      calc_actual_entropy(self.ds.df)

    for entropy_type in ['actS', 'hmpS']:
      # all
      x_train, x_test = train_test_split_entropy(self.ds.df, entropy_type, 'all', 0.2)
      classes = set(x_train[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
      classes = set(x_test[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
      # low
      x_train, x_test = train_test_split_entropy(self.ds.df, entropy_type, 'low', 0.2)
      classes = set(x_train[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low']))
      classes = set(x_test[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low']))
      # medium
      x_train, x_test = train_test_split_entropy(self.ds.df, entropy_type, 'medium', 0.2)
      classes = set(x_train[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['medium']))
      classes = set(x_test[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['medium']))
      # nohight
      x_train, x_test = train_test_split_entropy(self.ds.df, entropy_type, 'nohight', 0.2)
      classes = set(x_train[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low','medium']))
      classes = set(x_test[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['low','medium']))
      # hight
      x_train, x_test = train_test_split_entropy(self.ds.df, entropy_type, 'hight', 0.2)
      classes = set(x_train[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['hight']))
      classes = set(x_test[entropy_type+'_c'].unique())
      self.assertSequenceEqual(classes, set(['hight']))
