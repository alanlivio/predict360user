"""
Provides some prediction functions
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from .entropy import show_trajects_entropy, show_trajects_entropy_users


def get_train_test_split(df_trajects: pd.DataFrame, entropy: str,
                         perc_test: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  args = {'test_size': perc_test, 'random_state': 1}

  # default get 'all' entropy
  x_train, x_test = train_test_split(df_trajects, **args)

  # train
  if entropy.endswith('_users') and entropy != 'all':
    x_train, x_test = train_test_split(
        df_trajects[df_trajects['user_entropy_class'] ==
                    entropy.removesuffix('_users')], **args)
  elif entropy != 'all':
    x_train, x_test = train_test_split(
        df_trajects[df_trajects['traject_entropy_class'] == entropy],
        **args)

  return x_train, x_test


def show_train_test_split(df_trajects: pd.DataFrame, entropy: str, perc_test: float) -> None:
  x_train, x_test = get_train_test_split(df_trajects, entropy, perc_test)
  x_train['partition'] = 'train'
  x_test['partition'] = 'test'
  df_trajects = pd.concat([x_train, x_test])
  show_trajects_entropy(df_trajects, facet='partition')
  show_trajects_entropy_users(df_trajects, facet='partition')
