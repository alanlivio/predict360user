#!env python

import argparse
import logging
import os
import sys
from contextlib import redirect_stderr
from os.path import exists, join
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from users360 import (calc_trajects_entropy, config, dump_df_trajects,
                      get_traces, get_train_test_split)
from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

logging.basicConfig(level=logging.INFO, format='-- %(filename)s: %(message)s')

METRIC = all_metrics['orthodromic']
RATE = 0.2
BATCH_SIZE = 128.0

def create_model() -> Any:
  if MODEL_NAME == 'pos_only':
    from users360.head_motion_prediction.position_only_baseline import \
        create_pos_only_model
    return create_pos_only_model(M_WINDOW, H_WINDOW)
  else:
    raise NotImplementedError


def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch) -> np.array:
  positions_in_batch = np.array(positions_in_batch)
  eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2])
             for pos in batch] for batch in positions_in_batch]
  eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
  return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
  positions = positions * np.array([2 * np.pi, np.pi])
  eulerian_samples = [eulerian_to_cartesian(
    pos[0], pos[1]) for pos in positions]
  return np.array(eulerian_samples)


def generate_arrays(ids_l, future_window) -> Generator:
  while True:
    encoder_pos_inputs_for_batch = []
    # encoder_sal_inputs_for_batch = []
    decoder_pos_inputs_for_batch = []
    # decoder_sal_inputs_for_batch = []
    decoder_outputs_for_batch = []
    count = 0
    np.random.shuffle(ids_l)
    for ids in ids_l:
      user = ids['user']
      video = ids['video']
      x_i = ids['time-stamp']
      # Load the data
      if MODEL_NAME == 'pos_only':
        encoder_pos_inputs_for_batch.append(get_traces(
          video, user, DATASET_NAME)[x_i - M_WINDOW:x_i])
        decoder_pos_inputs_for_batch.append(
          get_traces(video, user, DATASET_NAME)[x_i:x_i + 1])
        decoder_outputs_for_batch.append(get_traces(video, user, DATASET_NAME)[
                         x_i + 1:x_i + future_window + 1])
      else:
        raise NotImplementedError
      count += 1
      if count == BATCH_SIZE:
        count = 0
        if MODEL_NAME == 'pos_only':
          yield ([
            transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
            transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)],
            transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
        else:
          raise NotImplementedError
        encoder_pos_inputs_for_batch = []
        # encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        # decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
    if count != 0:
      if MODEL_NAME == 'pos_only':
        yield ([
          transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
          transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)],
          transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
      else:
        raise NotImplementedError


def train() -> None:
  with redirect_stderr(open(os.devnull, 'w')): # pylint: disable=unspecified-encoding
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow.keras as keras

  steps_per_ep_train = np.ceil(len(PARTITION_IDS['train']) / BATCH_SIZE)
  steps_per_ep_validate = np.ceil(len(PARTITION_IDS['test']) / BATCH_SIZE)

  # train
  csv_logger_f = join(MODEL_FOLDER, 'train_results.csv')
  csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
  tb_callback = keras.callbacks.TensorBoard(log_dir=f'{MODEL_FOLDER}/logs')
  model_checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, mode='auto', period=1)
  if MODEL_NAME == 'pos_only':
    MODEL.fit_generator(
      generator=generate_arrays(PARTITION_IDS['train'], future_window=H_WINDOW),
      verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS,
      callbacks=[csv_logger, model_checkpoint, tb_callback],
      validation_data=generate_arrays(PARTITION_IDS['test'], future_window=H_WINDOW),
      validation_steps=steps_per_ep_validate
    )
  else:
    raise NotImplementedError


def evaluate() -> None:
  if MODEL_NAME == 'pos_only':
    MODEL.load_weights(MODEL_WEIGHTS)
  else:
    raise NotImplementedError
  errors_per_video = {}
  errors_per_timestep = {}

  for ids in tqdm(PARTITION_IDS['test'], desc='position predictions'):
    user = ids['user']
    video = ids['video']
    x_i = ids['time-stamp']

    # MODEL.predict
    if MODEL_NAME == 'pos_only':
      encoder_pos_inputs_for_sample = np.array([
        get_traces(video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]])
      decoder_pos_inputs_for_sample = np.array([
        get_traces(video, user, DATASET_NAME)[x_i:x_i + 1]])
    else:
      raise NotImplementedError

    groundtruth = get_traces(video, user, DATASET_NAME)[x_i + 1:x_i + H_WINDOW + 1]

    if MODEL_NAME == 'pos_only':
      model_pred = MODEL.predict([
        transform_batches_cartesian_to_normalized_eulerian(
          encoder_pos_inputs_for_sample),
        transform_batches_cartesian_to_normalized_eulerian(
          decoder_pos_inputs_for_sample)])[0]
      model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
    else:
      raise NotImplementedError

    if not video in errors_per_video:
      errors_per_video[video] = {}
    for t in range(len(groundtruth)):
      if t not in errors_per_video[video]:
        errors_per_video[video][t] = []
      errors_per_video[video][t].append(
        METRIC(groundtruth[t], model_prediction[t]))
      if t not in errors_per_timestep:
        errors_per_timestep[t] = []
      errors_per_timestep[t].append(
        METRIC(groundtruth[t], model_prediction[t]))

  result_basefilename = join(MODEL_FOLDER, TEST_PREFIX_PERC_ENTROPY)

  # avg_error_per_timestep
  avg_error_per_timestep = []
  for t in range(H_WINDOW):
    avg = np.mean(errors_per_timestep[t])
    avg_error_per_timestep.append(avg)
  # avg_error_per_timestep.csv
  result_file = f'{result_basefilename}_avg_error_per_timestep'
  logging.info(f'saving {result_file}.csv')
  np.savetxt(f'{result_file}.csv', avg_error_per_timestep)

  # avg_error_per_timestep.png
  plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep)
  met = 'orthodromic'
  plt.title(f'Average {met} in {DATASET_NAME} dataset using {MODEL_NAME} model')
  plt.ylabel(met)
  plt.xlim(2.5)
  plt.xlabel('Prediction step s (sec.)')
  logging.info(f'saving {result_file}.png')
  plt.savefig(result_file, bbox_inches='tight')

  # avg_error_per_video
  avg_error_per_video = []
  for video_name in VIDEOS_TEST:
    for t in range(H_WINDOW):
      if not video_name in errors_per_video:
        logging.error(f'missing {video_name} in VIDEOS_TEST')
        continue
      avg = np.mean(errors_per_video[video_name][t])
      avg_error_per_video.append(f'video={video_name} {t} {avg}')
  result_file = f'{result_basefilename}_avg_error_per_video.csv'
  np.savetxt(result_file, avg_error_per_video, fmt='%s')
  logging.info(f'saving {result_file}')


def compare_results() -> None:
  suffix = '_avg_error_per_timestep.csv'

  # find files with suffix
  dirs = [d for d in os.listdir(config.DATADIR) if d.startswith(MODEL_NAME)]
  csv_file_l = [(dir_name, file_name)
          for dir_name in dirs for file_name in os.listdir(join(config.DATADIR, dir_name))
          if (file_name.endswith(suffix) and file_name.startswith(TEST_PREFIX_PERC))]
  csv_data_l = [(dir_name, file_name, np.loadtxt(join(config.DATADIR, dir_name, file_name)))
          for (dir_name, file_name) in csv_file_l]
  assert csv_data_l, f'no data/<model>/{TEST_PREFIX_PERC}_*, run -evaluate'

  # sort by the last horizon hight
  # [2] is csv_data [-1] is last horizon
  def last_horizon_avg(item) -> int:
    return item[2][-1]
  csv_data_l.sort(reverse=True, key=last_horizon_avg)

  # plot as image
  for (dir_name, file_name, csv_data) in csv_data_l:
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, csv_data,
      label=f'{dir_name}_{file_name.removesuffix(suffix)}')
  met = 'orthodromic'
  plt.title(f'avg {met} (y) by pred. horizon (x) for {PERC_TEST} of dataset {DATASET_NAME}')
  plt.ylabel(met)
  plt.xlim(2.5)
  plt.xlabel('Prediction step s (sec.)')
  result_file = join(config.DATADIR, f'compare_{MODEL_NAME}')
  logging.info(f'saving {result_file}.png')
  plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
  plt.savefig(result_file, bbox_inches='tight')


if __name__ == '__main__':
  # argparse
  parser = argparse.ArgumentParser()
  parser.description = 'train or evaluate users360 models and datasets'
  model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
  entropy_l = ['all', 'low', 'medium', 'hight', # traject_entropy_class
     'low_users', 'medium_users','hight_users'] # user_entropy_class
  dataset_names = ['all', *config.DS_NAMES]

  # main actions params
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-load_raw_dataset', action='store_true',
             help='load raw dataset and save it as pickle ')
  group.add_argument('-calculate_entropy', action='store_true',
             help='calculate trajectories entropy')
  group.add_argument('-compare_results', action='store_true',
             help='show a comparison of -evaluate results')
  group.add_argument('-show_train_distribution', action='store_true',
            help='Show train distribution')
  group.add_argument('-train', action='store_true', help='Train model')
  group.add_argument('-evaluate', action='store_true', help='Evaluate model')

  # train only params
  parser.add_argument('-epochs', nargs='?', type=int, default=500,
            help='epochs numbers (default is 500)')
  parser.add_argument('-train_entropy', nargs='?', type=str, default='all',
            choices=entropy_l, help='entropy class to filter model used data  (default all)')

  # evaluate only params
  parser.add_argument('-test_entropy', nargs='?', type=str, default='all',
            choices=entropy_l, help='entropy class to filter -evalaute data (default all)')

  # train/evaluate params
  parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
            help='Used cuda gpu (default: 0)')
  parser.add_argument('-model_name', nargs='?', choices=model_names, default=model_names[0],
            help='reference model to used (default: pos_only)')
  parser.add_argument('-dataset_name', nargs='?', choices=dataset_names, default=dataset_names[0],
            help='dataset used to train this network  (default: all)')
  parser.add_argument('-init_window', nargs='?', type=int, default=30,
            help='initial buffer to avoid stationary part (default: 30)')
  parser.add_argument('-m_window', nargs='?', type=int, default=5,
            help='buffer window in timesteps (default: 5)')
  parser.add_argument('-h_window', nargs='?', type=int, default=25,
            help='''forecast window in timesteps (5 timesteps = 1 second)
                    used to predict (default: 25)''')
  parser.add_argument('-perc_test', nargs='?', type=float, default=0.2,
            help='test percetage (default: 0.2)')

  args = parser.parse_args()

  # dataset actions
  if args.load_raw_dataset:
    dump_df_trajects()
    sys.exit()
  if args.calculate_entropy:
    calc_trajects_entropy()
    dump_df_trajects()
    sys.exit()

  # used in next actions
  MODEL_NAME = args.model_name
  DATASET_NAME = args.dataset_name
  PERC_TEST = args.perc_test
  EPOCHS = args.epochs
  INIT_WINDOW = args.init_window
  M_WINDOW = args.m_window
  H_WINDOW = args.h_window
  END_WINDOW = H_WINDOW

  # MODEL_FOLDER
  train_entropy_suffix = '' if args.train_entropy == 'all' else f'_{args.train_entropy}_entropy'
  dataset_suffix = '' if args.dataset_name == 'all' else f'_{DATASET_NAME}'
  MODEL_FOLDER = join(config.DATADIR, f'{MODEL_NAME}{dataset_suffix}{train_entropy_suffix}')
  fmt_str = 'MODEL_NAME={}, dataset_suffix={}, train_entropy_suffix={}'
  logging.info(fmt_str.format(MODEL_NAME, dataset_suffix, train_entropy_suffix))
  logging.info(f'MODEL_FOLDER is {MODEL_FOLDER}')
  if not exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

  # MODEL_WEIGHTS
  MODEL_WEIGHTS = join(MODEL_FOLDER, 'weights.hdf5')
  logging.info(f'MODEL_WEIGHTS={MODEL_WEIGHTS}')
  if args.evaluate:
    assert exists(MODEL_WEIGHTS), f'{MODEL_WEIGHTS} does not exists'

  # TEST_PREFIX_PERC
  logging.info(f'PERC_TEST is {PERC_TEST}')
  TEST_PREFIX_PERC = f"test_{str(PERC_TEST).replace('.',',')}"
  TEST_PREFIX_PERC_ENTROPY = f'{TEST_PREFIX_PERC}_{args.test_entropy}'

  # compare_results action
  if args.compare_results:
    compare_results()
    sys.exit()

  # PARTITION_IDS for -train, -evaluation
  logging.info('')
  logging.info('partioning train/test ...')
  logging.info(f'x_train entropy is {args.train_entropy}')
  logging.info(f'x_test entropy is {args.test_entropy}')
  x_train, x_test = get_train_test_split(args.train_entropy, args.test_entropy, PERC_TEST)
  assert (not x_train.empty and not x_test.empty)

  logging.info(f'x_train has {len(x_train)} trajectories')
  logging.info(f'x_test has {len(x_test)} trajectories')
  PARTITION_IDS = {}
  PARTITION_IDS['train'] = [{'video': row[1]['ds_video'],
              'user': row[1]['ds_user'], 'time-stamp': tstap}
              for row in x_train.iterrows()
              for tstap in range(INIT_WINDOW, row[1]['traject'].shape[0] - END_WINDOW)]
  PARTITION_IDS['test'] = [{'video': row[1]['ds_video'],
             'user': row[1]['ds_user'], 'time-stamp': tstap}
             for row in x_test.iterrows()
             for tstap in range(INIT_WINDOW, row[1]['traject'].shape[0] - END_WINDOW)]
  VIDEOS_TEST = x_test['ds_video'].unique()
  USERS_TEST = x_test['ds_user'].unique()
  logging.info(f"PARTITION_IDS['train'] has {len(PARTITION_IDS['train'])} position predictions")
  logging.info(f"PARTITION_IDS['test'] has {len(PARTITION_IDS['test'])} position predictions")

  # creating model
  logging.info('')
  logging.info('creating model ...')
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  if args.gpu_id:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
  MODEL = create_model()

  # train, evaluate actions
  if args.train:
    logging.info('')
    logging.info('training ...')
    logging.info(f'EPOCHS is {EPOCHS}')
    train()
  elif args.evaluate:
    logging.info('')
    logging.info('evaluating ...')
    logging.info(f'evaluate_entropy is {args.train_entropy}')
    evaluate()
  sys.exit()
