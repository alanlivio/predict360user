#!env python

import argparse
import sys

from users360 import config
from users360.entropy import calc_trajects_entropy
from users360.trainer import Trainer, compare_results
from users360.trajects import *

if __name__ == '__main__':
  # argparse
  psr = argparse.ArgumentParser()
  psr.description = 'train or evaluate users360 models and datasets'

  # actions params
  grp = psr.add_mutually_exclusive_group()
  grp.add_argument('-calculate_entropy',
                   action='store_true',
                   help='load dataset, calculate entropy and save it ')
  grp.add_argument('-compare_results', action='store_true', help='compare -evaluate results ')
  grp.add_argument('-train', action='store_true', help='train model')
  grp.add_argument('-evaluate', action='store_true', help='evaluate model')

  # Trainer params
  psr.add_argument('-gpu_id', nargs='?', type=int, default=0, help='Used cuda gpu (default: 0)')
  psr.add_argument('-model_name',
                   nargs='?',
                   choices=config.MODEL_NAMES,
                   default=config.MODEL_NAMES[0],
                   help='reference model to used (default: pos_only)')
  psr.add_argument('-dataset_name',
                   nargs='?',
                   choices=config.ARGS_DS_NAMES,
                   default=config.ARGS_DS_NAMES[0],
                   help='dataset used to train this network  (default: all)')
  psr.add_argument('-init_window',
                   nargs='?',
                   type=int,
                   default=30,
                   help='initial buffer to avoid stationary part (default: 30)')
  psr.add_argument('-m_window',
                   nargs='?',
                   type=int,
                   default=5,
                   help='buffer window in timesteps (default: 5)')
  psr.add_argument('-h_window',
                   nargs='?',
                   type=int,
                   default=25,
                   help='''forecast window in timesteps (5 timesteps = 1 second)
                           used to predict (default: 25)''')
  psr.add_argument('-perc_test',
                   nargs='?',
                   type=float,
                   default=0.2,
                   help='test percetage (default: 0.2)')
  psr.add_argument('-dry_run',
                   action='store_true',
                   help='show train/test info but stop before perform')
  psr.add_argument('-train_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=config.ARGS_ENTROPY_NAMES + config.ARGS_ENTROPY_AUTO_NAMES,
                   help='''entropy to filter train data (default all).
                           -evaluate accepts auto, auto_m_window, auto_since_start''')

  # Trainer.train() only params
  psr.add_argument('-epochs',
                   nargs='?',
                   type=int,
                   default=100,
                   help='epochs numbers (default is 500)')

  # Trainer.evaluate() only params
  psr.add_argument('-test_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=config.ARGS_ENTROPY_NAMES,
                   help='entropy to filter test data (default all)')
  psr.add_argument('-test_user', nargs='?', default='', type=str, help='user to filter test data')
  psr.add_argument('-test_video', nargs='?', default='', type=str, help='video to filter test data')

  args = psr.parse_args()

  # -calculate_entropy
  if args.calculate_entropy:
    df_tmp = get_df_trajects()
    calc_trajects_entropy(df_tmp)
    dump_df_trajects(df_tmp)
  # -compare_results
  elif args.compare_results:
    compare_results(args.model_name, args.perc_test)
  else:
    trn_args = {
        'dataset_name': args.dataset_name,
        'model_name': args.model_name,
        'perc_test': args.perc_test,
        'init_window': args.init_window,
        'm_window': args.m_window,
        'h_window': args.h_window,
        'dry_run': args.dry_run,
        'epochs': args.epochs,
        'train_entropy': args.train_entropy,
        'test_entropy': args.test_entropy,
        'test_user': args.test_user,
        'test_video': args.test_video
    }
    trn = Trainer(**trn_args)
    if args.train:
      trn.train()
    elif args.evaluate:
      trn.evaluate()
  sys.exit()