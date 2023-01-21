import unittest
from os.path import join

from users360 import config
from users360.trainer import Trainer, count_traject_entropy_classes


class Test(unittest.TestCase):

  def test_init(self) -> None:
    trn = Trainer()
    assert trn.model_dir == join(config.DATADIR, 'pos_only')
    for name in config.ARGS_ENTROPY_NAMES[1:]:
      trn = Trainer(train_entropy=name)
      assert trn.model_fullname == f'pos_only_{name}_entropy'
      assert trn.model_dir == join(config.DATADIR, f'pos_only_{name}_entropy')
      assert trn.test_res_basename == join(trn.model_dir,f'test_0,2_all')
      assert trn.evaluate_auto == False
      trn = Trainer(train_entropy=name, dataset_name='david')
      assert trn.model_fullname == f'pos_only_david_{name}_entropy'
      assert trn.model_dir == join(config.DATADIR, f'pos_only_david_{name}_entropy')
      assert trn.test_res_basename == join(trn.model_dir,f'test_0,2_all')
      assert trn.evaluate_auto == False
    for name in ['low','medium','hight']:
      trn = Trainer(test_entropy=name)
      assert trn.test_res_basename == join(trn.model_dir,f'test_0,2_{name}')
      assert trn.evaluate_auto == False
    for name in config.ARGS_ENTROPY_AUTO_NAMES:
      trn = Trainer(test_entropy=name)
      assert trn.test_res_basename == join(trn.model_dir,f'test_0,2_{name}')
      assert trn.evaluate_auto == True

  def test_partition_train(self) -> None:
    trn = Trainer(dry_run=True)
    trn.partition_train()
    assert (count_traject_entropy_classes(trn.x_train) == (10301, 6125, 3133, 1043))
    trn = Trainer(train_entropy='low')
    trn.partition_train()
    assert (count_traject_entropy_classes(trn.x_train) == (6108, 6108, 0, 0))
    trn = Trainer(train_entropy='medium')
    trn.partition_train()
    assert (count_traject_entropy_classes(trn.x_train) == (3157, 0, 3157, 0))
    trn = Trainer(train_entropy='hight')
    trn.partition_train()
    assert (count_traject_entropy_classes(trn.x_train) == (1036, 0, 0, 1036))

  def test_partition_evaluate(self) -> None:
    trn = Trainer(dry_run=True)
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (2576, 1510, 814, 252))

    trn = Trainer( test_entropy='low')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (1527, 1527, 0, 0))

    trn = Trainer( test_entropy='medium')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (790, 0, 790, 0))

    trn = Trainer( test_entropy='hight')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (259, 0, 0, 259))