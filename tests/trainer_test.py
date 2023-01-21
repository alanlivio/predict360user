import unittest
from os.path import join

from users360 import config
from users360.entropy import calc_actual_entropy
from users360.trainer import (Trainer, count_traject_entropy_classes,
                              get_train_test_split)
from users360.trajects import get_df_trajects


class Test(unittest.TestCase):

  def test_init(self) -> None:
    trn = Trainer()
    assert trn.model_fullname == 'pos_only'
    assert trn.model_dir == join(config.DATADIR, 'pos_only')
    assert trn.test_res_basename == join(trn.model_dir, 'test_0,2_all')
    assert trn.evaluate_auto is False
    trn = Trainer(dataset_name='david')
    assert trn.model_fullname == 'pos_only_david'
    assert trn.evaluate_auto is False
    for train_name in config.ARGS_ENTROPY_NAMES[1:]:
      for test_entropy in config.ARGS_ENTROPY_NAMES[1:]:
        trn = Trainer(train_entropy=train_name, test_entropy=test_entropy)
        assert trn.model_fullname == f'pos_only_{train_name}_entropy'
        assert trn.model_dir == join(config.DATADIR, f'pos_only_{train_name}_entropy')
        assert trn.test_res_basename == join(trn.model_dir, f'test_0,2_{test_entropy}')
        assert trn.evaluate_auto is False
    for train_name in config.ARGS_ENTROPY_AUTO_NAMES:
      for test_entropy in ['low', 'medium', 'hight']: # only supported for auto train
        trn = Trainer(train_entropy=train_name, test_entropy=test_entropy)
        assert trn.model_fullname == f'pos_only_{train_name}_entropy'
        assert trn.model_dir == join(config.DATADIR, f'pos_only_{train_name}_entropy')
        assert trn.test_res_basename == join(trn.model_dir, f'test_0,2_{test_entropy}')
        assert trn.evaluate_auto is True

  def test_train_test_split(self) -> None:
    self.df = get_df_trajects()
    assert not self.df.empty

    if not 'traject_entropy_class' in self.df.columns:
      calc_actual_entropy(self.df)

    # x_train all
    x_train, x_test = get_train_test_split(self.df, 'all', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])

    # x_train low
    x_train, x_test = get_train_test_split(self.df, 'low', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['low'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['low'])

    # x_train medium
    x_train, x_test = get_train_test_split(self.df, 'medium', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['medium'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['medium'])

    # x_train hight
    x_train, x_test = get_train_test_split(self.df, 'hight', 0.2)
    unique_s = set(x_train['traject_entropy_class'].unique())
    assert unique_s == set(['hight'])
    unique_s = set(x_test['traject_entropy_class'].unique())
    assert unique_s == set(['hight'])

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

    trn = Trainer(test_entropy='low')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (1527, 1527, 0, 0))

    trn = Trainer(test_entropy='medium')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (790, 0, 790, 0))

    trn = Trainer(test_entropy='hight')
    trn.partition_evaluate()
    assert (count_traject_entropy_classes(trn.x_test) == (259, 0, 0, 259))