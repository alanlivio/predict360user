import unittest
from os.path import join

from predict360user import config
from predict360user.trainer import Trainer


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

  def test_train_partition(self) -> None:
    trn = Trainer(train_entropy='all')
    # all
    trn.train_entropy = 'all'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # low
    trn.train_entropy = 'low'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # medium
    trn.train_entropy = 'medium'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # nolow
    trn.train_entropy = 'nolow'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium','hight']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['medium','hight']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # nohight
    trn.train_entropy = 'nohight'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low','medium']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
    # hight
    trn.train_entropy = 'hight'
    trn._train_partition()
    self.assertGreater(len(trn.x_train), len(trn.x_val))
    classes = set(trn.x_train['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['hight']))
    classes = set(trn.x_val['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['hight']))
    classes = set(trn.x_test['actS_c'].unique())
    self.assertSequenceEqual(classes, set(['low', 'medium', 'hight']))
