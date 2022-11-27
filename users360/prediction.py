"""
Provides some prediction functions
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .entropy import show_trajects_entropy, show_trajects_entropy_users
from .trajects import get_df_trajects


def get_train_test_split(train_entropy, test_entropy,
                         perc_test) -> tuple[pd.DataFrame, pd.DataFrame]:
  df = get_df_trajects()
  if (train_entropy != 'all') and (test_entropy != 'all'):
    assert not df[
        'traject_entropy_class'].empty, "no 'traject_entropy_class', run -calc_trajects_entropy"
    x_train, x_test = train_test_split(df, test_size=perc_test, random_state=1)
  else:
    if train_entropy == 'all':
      x_train, _ = train_test_split(df, test_size=perc_test, random_state=1)
    else:
      x_train, _ = train_test_split(df[df['traject_entropy_class'] == train_entropy],
                                    test_size=perc_test,
                                    random_state=1)
    if test_entropy == 'all':
      _, x_test = train_test_split(df, test_size=perc_test, random_state=1)
    else:
      _, x_test = train_test_split(df[df['traject_entropy_class'] == test_entropy],
                                   test_size=perc_test,
                                   random_state=1)
  return x_train, x_test


def show_train_test_split(train_entropy, test_entropy, perc_test) -> None:
  x_train, x_test = get_train_test_split(train_entropy, test_entropy, perc_test)
  x_train['partition'] = 'train'
  x_test['partition'] = 'test'
  config.df_trajects = pd.concat([x_train, x_test])
  show_trajects_entropy(facet='partition')
  show_trajects_entropy_users(facet='partition')
