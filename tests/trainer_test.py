import unittest
from os.path import join

from predict360user.trainer import Trainer
from predict360user.utils import config


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

  def test_train(self) -> None:
    trn = Trainer(train_entropy='all', dry_run=True) # dry_run stop before build model
    # all
    trn.train_entropy = 'all'
    trn.train() 
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # low
    trn.train_entropy = 'low'
    trn.train() 
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # medium
    trn.train_entropy = 'medium'
    trn.train() 
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # nohight
    trn.train_entropy = 'nohight'
    trn.train() 
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # hight
    trn.train_entropy = 'hight'
    trn.train() 
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['hight']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['hight']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
