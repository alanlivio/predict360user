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
  args = {'test_size': perc_test, 'random_state':1}
  
  # default get 'all' entropy
  x_train, x_test = train_test_split(df, **args)
  
  # train
  if train_entropy.endswith('_users') and train_entropy != 'all':
    x_train, _ = train_test_split( 
      df[df['user_entropy_class'] == train_entropy.removesuffix('_users')], **args)
  elif train_entropy != 'all':
    x_train, _ = train_test_split(
      df[df['traject_entropy_class'] == train_entropy], **args)
  
  # test
  if test_entropy.endswith('_users') and test_entropy != 'all':
    _, test_entropy = train_test_split( 
      df[df['user_entropy_class'] == test_entropy.removesuffix('_users')], **args)
  elif test_entropy != 'all':
    _, x_test = train_test_split(
      df[df['traject_entropy_class'] == test_entropy], **args)
  
  return x_train, x_test


def show_train_test_split(train_entropy, test_entropy, perc_test) -> None:
  x_train, x_test = get_train_test_split(train_entropy, test_entropy, perc_test)
  x_train['partition'] = 'train'
  x_test['partition'] = 'test'
  config.df_trajects = pd.concat([x_train, x_test])
  show_trajects_entropy(facet='partition')
  show_trajects_entropy_users(facet='partition')
