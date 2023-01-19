#!env python

import argparse
import sys

from . import config
from .entropy import calc_trajects_entropy
from .evaluator import Evaluator, compare_results
from .trainer import Trainer
from .trajects import *

if __name__ == '__main__':
  # argparse
  psr = argparse.ArgumentParser()
  psr.description = 'train or evaluate users360 models and datasets'

  # main actions params
  grp = psr.add_mutually_exclusive_group()
  grp.add_argument('-calculate_entropy',
                   action='store_true',
                   help='load dataset, calculate entropy and save it ')
  grp.add_argument('-compare_results', action='store_true', help='compare -evaluate results ')
  grp.add_argument('-train', action='store_true', help='train model')
  grp.add_argument('-evaluate', action='store_true', help='evaluate model')

  # train only params
  psr.add_argument('-epochs',
                   nargs='?',
                   type=int,
                   default=100,
                   help='epochs numbers (default is 500)')

  psr.add_argument('-train_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=config.ARGS_ENTROPY_NAMES,
                   help='entropy to filter data model train  (default all)')

  # evaluate only params
  test_model_l = config.ARGS_ENTROPY_NAMES + ['auto', 'auto_m_window', 'auto_since_start']
  psr.add_argument('-test_model_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=test_model_l,
                   help='''entropy of the model to be used.
                          auto selects from traject entropy.
                          auto_window selects from last window''')
  psr.add_argument('-test_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=config.ARGS_ENTROPY_NAMES,
                   help='entropy class to filter -evaluate data (default all)')
  psr.add_argument('-oneuser',
                   nargs='?',
                   type=str,
                   help='one user for evaluation')
  psr.add_argument('-onevideo',
                   nargs='?',
                   type=str,
                   help='one video for evaluation')

  # train/evaluate params
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

  args = psr.parse_args()

  # global vars
  cfg = {}
  cfg['dataset_name'] = args.dataset_name
  cfg['model_name'] = args.model_name
  cfg['perc_test'] = args.perc_test
  cfg['epochs'] = args.epochs
  cfg['init_window'] = args.init_window
  cfg['m_window'] = args.m_window
  cfg['h_window'] = args.h_window
  cfg['test_prefix_perc'] = f"test_{str(args.perc_test).replace('.',',')}"
  cfg['test_model_entropy'] = args.test_model_entropy
  cfg['evaluate_auto'] = args.test_model_entropy.startswith('auto')
  cfg['train_entropy'] = args.train_entropy
  cfg['test_entropy'] = args.test_entropy
  cfg['oneuser'] = args.oneuser
  cfg['onevideo'] = args.onevideo
  cfg['dry_run'] = args.dry_run
  if args.gpu_id:
    cfg['gpu_id'] = str(args.gpu_id)

  # -calculate_entropy
  if args.calculate_entropy:
    df_tmp = get_df_trajects()
    calc_trajects_entropy(df_tmp)
    dump_df_trajects(df_tmp)
  # -compare_results
  elif args.compare_results:
    compare_results(args.model_name, args.perc_test)
  # -train
  elif args.train:
    Trainer(cfg).train()
  # -evaluate
  elif args.evaluate:
    Evaluator(cfg).evaluate()
  sys.exit()