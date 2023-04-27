#!env python

import argparse
import sys

from predict360user import Trainer, config

if __name__ == '__main__':
  # argparse
  psr = argparse.ArgumentParser()
  psr.description = 'train or evaluate users360 models and datasets'

  # actions params
  grp = psr.add_mutually_exclusive_group(required=True)
  grp.add_argument('-train', action='store_true', help='train model')
  grp.add_argument('-compare_train', action='store_true', help='compare -train results')
  grp.add_argument('-evaluate', action='store_true', help='evaluate model')
  grp.add_argument('-compare_evaluate', action='store_true', help='compare -evaluate results')

  # Trainer params
  psr.add_argument('-model_name',
                   nargs='?',
                   choices=config.ARGS_MODEL_NAMES,
                   default=config.ARGS_MODEL_NAMES[0],
                   help='reference model to used (default: pos_only)')
  psr.add_argument('-dataset_name',
                   nargs='?',
                   choices=config.ARGS_DS_NAMES,
                   default=config.ARGS_DS_NAMES[0],
                   help='dataset used to train this network (default: all)')
  psr.add_argument('-train_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=config.ARGS_ENTROPY_NAMES + config.ARGS_ENTROPY_AUTO_NAMES,
                   help='''entropy to filter train data (default all).
                           -evaluate accepts auto, auto_m_window, auto_since_start''')
  psr.add_argument('-dry_run',
                  action='store_true',
                  help='show train/test info but stop before perform')

  # psr.add_argument('-init_window',
  #                  nargs='?',
  #                  type=int,
  #                  default=30,
  #                  help='initial buffer to avoid stationary part (default: 30)')
  # psr.add_argument('-m_window',
  #                  nargs='?',
  #                  type=int,
  #                  default=5,
  #                  help='buffer window in timesteps (default: 5)')
  # psr.add_argument('-h_window',
  #                  nargs='?',
  #                  type=int,
  #                  default=25,
  #                  help='''forecast window in timesteps (5 timesteps = 1 second)
  #                          used to predict (default: 25)''')
  psr.add_argument('-test_size',
                   nargs='?',
                   type=float,
                   default=0.2,
                   help='test percetage (default: 0.2)')

  # train() only params
  psr.add_argument('-gpu_id',
                   nargs='?',
                   type=int,
                   default=0,
                   help=f'gpu_id to be used (default is 0)')
  psr.add_argument('-epochs',
                   nargs='?',
                   type=int,
                   default=config.DEFAULT_EPOCHS,
                   help=f'epochs numbers (default is {config.DEFAULT_EPOCHS})')

  args = psr.parse_args()
  trn_args = {
      'dataset_name': args.dataset_name,
      'model_name': args.model_name,
      'train_entropy': args.train_entropy,
      'dry_run': args.dry_run,
      'epochs': args.epochs,
      'test_size': args.test_size,
      # 'init_window': args.init_window,
      # 'm_window': args.m_window,
      # 'h_window': args.h_window,
  }
  trn = Trainer(**trn_args)
  if args.train:
    trn.train()
  elif args.evaluate:
    trn.evaluate()
  elif args.compare_train:
    trn.compare_train()
  elif args.compare_evaluate:
    trn.compare_evaluate()
  sys.exit()