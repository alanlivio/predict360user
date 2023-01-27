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
    assert trn.model_fullname == 'pos_only,all,,'
    assert trn.model_dir == join(config.DATADIR, trn.model_fullname)
    assert trn.using_auto is False
    trn = Trainer(dataset_name='david')
    assert trn.model_fullname == 'pos_only,david,,'
    assert trn.using_auto is False
    for train_entropy in config.ARGS_ENTROPY_NAMES[1:] + config.ARGS_ENTROPY_AUTO_NAMES:
        trn = Trainer(train_entropy=train_entropy)
        entropy_type = 'hmpS' if train_entropy.endswith('hmp') else 'actS'
        train_entropy = train_entropy.removesuffix('hmp')
        model_fullname = f'pos_only,all,{entropy_type},{train_entropy}'
        assert trn.model_fullname == model_fullname
        assert trn.model_dir == join(config.DATADIR, model_fullname)
        using_auto = train_entropy.startswith('auto')
        assert trn.using_auto is using_auto

  def test_train_test_split(self) -> None:
    self.df = get_df_trajects()
    assert not self.df.empty

    if not 'actS_c' in self.df.columns:
      calc_actual_entropy(self.df)

    # x_train all
    x_train, x_test = get_train_test_split(self.df, 'all', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])
    unique_s = set(x_test['actS_c'].unique())
    assert unique_s == set(['low', 'medium', 'hight'])

    # x_train low
    x_train, x_test = get_train_test_split(self.df, 'low', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    assert unique_s == set(['low'])
    unique_s = set(x_test['actS_c'].unique())
    assert unique_s == set(['low'])

    # x_train medium
    x_train, x_test = get_train_test_split(self.df, 'medium', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    assert unique_s == set(['medium'])
    unique_s = set(x_test['actS_c'].unique())
    assert unique_s == set(['medium'])

    # x_train hight
    x_train, x_test = get_train_test_split(self.df, 'hight', 0.2)
    unique_s = set(x_train['actS_c'].unique())
    assert unique_s == set(['hight'])
    unique_s = set(x_test['actS_c'].unique())
    assert unique_s == set(['hight'])

  def test_partition(self) -> None:
    trn = Trainer()
    trn.partition()
    assert (count_traject_entropy_classes(trn.x_train) == (10301, 6125, 3133, 1043))
    assert (count_traject_entropy_classes(trn.x_test) == (2576, 1510, 814, 252))
    trn = Trainer(train_entropy='low')
    trn.partition()
    assert (count_traject_entropy_classes(trn.x_train) == (6108, 6108, 0, 0))
    assert (count_traject_entropy_classes(trn.x_test) == (1527, 1527, 0, 0))
    trn = Trainer(train_entropy='medium')
    trn.partition()
    assert (count_traject_entropy_classes(trn.x_train) == (3157, 0, 3157, 0))
    assert (count_traject_entropy_classes(trn.x_test) == (790, 0, 790, 0))
    trn = Trainer(train_entropy='hight')
    trn.partition()
    assert (count_traject_entropy_classes(trn.x_train) == (1036, 0, 0, 1036))
    assert (count_traject_entropy_classes(trn.x_test) == (259, 0, 0, 259))
    trn = Trainer(train_entropy='nohight')
    trn.partition()
    assert (count_traject_entropy_classes(trn.x_train) == (9265, 6089, 3176, 0))
    assert (count_traject_entropy_classes(trn.x_test) == (2317, 1546, 771, 0))