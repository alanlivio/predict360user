#!env python

import argparse
import os
import sys
from contextlib import redirect_stderr
from os.path import exists, join
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

from users360 import (calc_actual_entropy, calc_trajects_entropy, config,
                      dump_df_trajects, get_class_by_threshold,
                      get_df_trajects, get_traces, get_train_test_split,
                      get_trajects_entropy_threshold)
from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

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


def transform_batches_cartesian_to_normalized_eulerian(
    positions_in_batch) -> np.array:
  positions_in_batch = np.array(positions_in_batch)
  eulerian_batches = [[
      cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch
  ] for batch in positions_in_batch]
  eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
  return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
  positions = positions * np.array([2 * np.pi, np.pi])
  eulerian_samples = [
      eulerian_to_cartesian(pos[0], pos[1]) for pos in positions
  ]
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
      x_i = ids['trace_id']
      # Load the data
      if MODEL_NAME == 'pos_only':
        encoder_pos_inputs_for_batch.append(
            get_traces(DF_TRAJECTS, video, user,
                       DATASET_NAME)[x_i - M_WINDOW:x_i])
        decoder_pos_inputs_for_batch.append(
            get_traces(DF_TRAJECTS, video, user, DATASET_NAME)[x_i:x_i + 1])
        decoder_outputs_for_batch.append(
            get_traces(DF_TRAJECTS, video, user,
                       DATASET_NAME)[x_i + 1:x_i + future_window + 1])
      else:
        raise NotImplementedError
      count += 1
      if count == BATCH_SIZE:
        count = 0
        if MODEL_NAME == 'pos_only':
          yield ([
              transform_batches_cartesian_to_normalized_eulerian(
                  encoder_pos_inputs_for_batch),
              transform_batches_cartesian_to_normalized_eulerian(
                  decoder_pos_inputs_for_batch)
          ],
                 transform_batches_cartesian_to_normalized_eulerian(
                     decoder_outputs_for_batch))
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
            transform_batches_cartesian_to_normalized_eulerian(
                encoder_pos_inputs_for_batch),
            transform_batches_cartesian_to_normalized_eulerian(
                decoder_pos_inputs_for_batch)
        ],
               transform_batches_cartesian_to_normalized_eulerian(
                   decoder_outputs_for_batch))
      else:
        raise NotImplementedError


def train() -> None:
  with redirect_stderr(open(os.devnull, 'w')):  # pylint: disable=unspecified-encoding
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow.keras as keras

  steps_per_ep_train = np.ceil(len(PARTITION_IDS['train']) / BATCH_SIZE)
  steps_per_ep_validate = np.ceil(len(PARTITION_IDS['test']) / BATCH_SIZE)

  # train
  csv_logger_f = join(MODEL_FOLDER, 'train_results.csv')
  csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
  tb_callback = keras.callbacks.TensorBoard(log_dir=f'{MODEL_FOLDER}/logs')
  model_checkpoint = keras.callbacks.ModelCheckpoint(MODEL_WEIGHTS,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)
  if MODEL_NAME == 'pos_only':
    MODEL.fit_generator(generator=generate_arrays(PARTITION_IDS['train'],
                                                  future_window=H_WINDOW),
                        verbose=1,
                        steps_per_epoch=steps_per_ep_train,
                        epochs=EPOCHS,
                        callbacks=[csv_logger, model_checkpoint, tb_callback],
                        validation_data=generate_arrays(PARTITION_IDS['test'],
                                                        future_window=H_WINDOW),
                        validation_steps=steps_per_ep_validate)
  else:
    raise NotImplementedError


def evaluate() -> None:
  if MODEL_NAME == 'pos_only':
    if EVALUATE_AUTO:
      model_low = create_model()
      model_low.load_weights(MODEL_WEIGHTS_LOW)
      model_medium = create_model()
      model_medium.load_weights(MODEL_WEIGHTS_MEDIUM)
      model_hight = create_model()
      model_hight.load_weights(MODEL_WEIGHTS_HIGHT)
      threshold_medium, threshold_hight = get_trajects_entropy_threshold(DF_TRAJECTS)
    else:
      MODEL.load_weights(MODEL_WEIGHTS)
  else:
    raise NotImplementedError
  errors_per_video = {}
  errors_per_timestep = {}

  for ids in tqdm(PARTITION_IDS['test'], desc='position predictions'):
    user = ids['user']
    video = ids['video']
    x_i = ids['trace_id']

    # MODEL.predict
    if MODEL_NAME == 'pos_only':
      encoder_pos_inputs_for_sample = np.array([
          get_traces(DF_TRAJECTS, video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]
      ])
      decoder_pos_inputs_for_sample = np.array(
          [get_traces(DF_TRAJECTS, video, user, DATASET_NAME)[x_i:x_i + 1]])
    else:
      raise NotImplementedError

    groundtruth = get_traces(DF_TRAJECTS, video, user,
                             DATASET_NAME)[x_i + 1:x_i + H_WINDOW + 1]

    if MODEL_NAME == 'pos_only':
      current_model = MODEL
      if EVALUATE_AUTO:
        if TEST_MODEL_ENTROPY == 'auto':
          traject_entropy_class = ids['traject_entropy_class']
        if TEST_MODEL_ENTROPY == 'auto_m_window':
          window = get_traces(DF_TRAJECTS, video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        elif TEST_MODEL_ENTROPY == 'auto_since_start':
          window = get_traces(DF_TRAJECTS, video, user, DATASET_NAME)[0:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        else:
          raise RuntimeError()
        if traject_entropy_class == 'low':
          current_model = model_low
        elif traject_entropy_class == 'medium':
          current_model = model_medium
        elif traject_entropy_class == 'hight':
          current_model = model_hight
        else:
          raise NotImplementedError
      model_pred = current_model.predict([
          transform_batches_cartesian_to_normalized_eulerian(
              encoder_pos_inputs_for_sample),
          transform_batches_cartesian_to_normalized_eulerian(
              decoder_pos_inputs_for_sample)
      ])[0]
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
      errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

  result_basefilename = join(MODEL_FOLDER, EVALUATE_NAME)

  # avg_error_per_timestep
  avg_error_per_timestep = []
  for t in range(H_WINDOW):
    avg = np.mean(errors_per_timestep[t])
    avg_error_per_timestep.append(avg)
  # avg_error_per_timestep.csv
  result_file = f'{result_basefilename}_avg_error_per_timestep'
  config.loginf(f'saving {result_file}.csv')
  np.savetxt(f'{result_file}.csv', avg_error_per_timestep)

  # avg_error_per_timestep.png
  plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep)
  met = 'orthodromic'
  plt.title(f'Average {met} in {DATASET_NAME} dataset using {MODEL_NAME} model')
  plt.ylabel(met)
  plt.xlim(2.5)
  plt.xlabel('Prediction step s (sec.)')
  config.loginf(f'saving {result_file}.png')
  plt.savefig(result_file, bbox_inches='tight')

  # avg_error_per_video
  avg_error_per_video = []
  for video_name in VIDEOS_TEST:
    for t in range(H_WINDOW):
      if not video_name in errors_per_video:
        config.logerr(f'missing {video_name} in VIDEOS_TEST')
        continue
      avg = np.mean(errors_per_video[video_name][t])
      avg_error_per_video.append(f'video={video_name} {t} {avg}')
  result_file = f'{result_basefilename}_avg_error_per_video.csv'
  np.savetxt(result_file, avg_error_per_video, fmt='%s')
  config.loginf(f'saving {result_file}')


def compare_results() -> None:
  suffix = '_avg_error_per_timestep.csv'

  # find files with suffix
  dirs = [d for d in os.listdir(config.DATADIR) if d.startswith(MODEL_NAME)]
  csv_file_l = [
      (dir_name, file_name) for dir_name in dirs
      for file_name in os.listdir(join(config.DATADIR, dir_name))
      if (file_name.endswith(suffix) and file_name.startswith(TEST_PREFIX_PERC))
  ]
  csv_data_l = [ (f'{dir_name}_{file_name.removesuffix(suffix)}', horizon, error)
                for (dir_name, file_name) in csv_file_l
                for horizon, error in enumerate(np.loadtxt(join(config.DATADIR, dir_name, file_name)))
              ]
  assert csv_data_l, f'no data/<model>/{TEST_PREFIX_PERC}_*, run -evaluate'

  # plot image
  df_compare = pd.DataFrame(csv_data_l, columns=['name', 'horizon', 'vidoes_avg_error'])
  df_compare = df.sort_values(ascending=False, by="vidoes_avg_error")
  fig = px.line(df_compare, x='horizon', y="vidoes_avg_error", color='name', color_discrete_sequence=px.colors.qualitative.G10)
  result_file = join(config.DATADIR, f'compare_{MODEL_NAME}.png')
  config.loginf(f'saving {result_file}')
  fig.write_image(result_file)


if __name__ == '__main__':
  # argparse
  parser = argparse.ArgumentParser()
  parser.description = 'train or evaluate users360 models and datasets'
  model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
  entropy_l = [
      'all',
      'low',
      'medium',
      'hight',
      'low_users',
      'medium_users',
      'hight_users',
  ]
  dataset_names = ['all', *config.DS_NAMES]

  # main actions params
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      '-calculate_entropy',
      action='store_true',
      help='load raw dataset, calculate entropy and save it as pickle')
  group.add_argument('-compare_results',
                     action='store_true',
                     help='compare -evaluate results ')
  group.add_argument('-train', action='store_true', help='train model')
  group.add_argument('-evaluate', action='store_true', help='evaluate model')
  group.add_argument(
      '-evaluate_adaptative',
      action='store_true',
      help='''Evaluate te test choosing using the {low,medium, hight}
                    model for the respective traject entropy''')

  # train only params
  parser.add_argument('-epochs',
                      nargs='?',
                      type=int,
                      default=100,
                      help='epochs numbers (default is 500)')

  parser.add_argument('-train_entropy',
                      nargs='?',
                      type=str,
                      default='all',
                      choices=entropy_l,
                      help='entropy to filter data model train  (default all)')

  # evaluate only params
  test_model_l = entropy_l + ['auto', 'auto_m_window', 'auto_since_start']
  parser.add_argument(
      '-test_model_entropy',
      nargs='?',
      type=str,
      default='all',
      choices=test_model_l,
      help='''entropy of the model to be used, auto selects from traject entropy,
               and auto_window selects from last window''')
  parser.add_argument(
      '-test_entropy',
      nargs='?',
      type=str,
      default='all',
      choices=entropy_l,
      help='entropy class to filter -evaluate data (default all)')

  # train/evaluate params
  parser.add_argument('-gpu_id',
                      nargs='?',
                      type=int,
                      default=0,
                      help='Used cuda gpu (default: 0)')
  parser.add_argument('-model_name',
                      nargs='?',
                      choices=model_names,
                      default=model_names[0],
                      help='reference model to used (default: pos_only)')
  parser.add_argument('-dataset_name',
                      nargs='?',
                      choices=dataset_names,
                      default=dataset_names[0],
                      help='dataset used to train this network  (default: all)')
  parser.add_argument(
      '-init_window',
      nargs='?',
      type=int,
      default=30,
      help='initial buffer to avoid stationary part (default: 30)')
  parser.add_argument('-m_window',
                      nargs='?',
                      type=int,
                      default=5,
                      help='buffer window in timesteps (default: 5)')
  parser.add_argument(
      '-h_window',
      nargs='?',
      type=int,
      default=25,
      help='''forecast window in timesteps (5 timesteps = 1 second)
                    used to predict (default: 25)''')
  parser.add_argument('-perc_test',
                      nargs='?',
                      type=float,
                      default=0.2,
                      help='test percetage (default: 0.2)')

  args = parser.parse_args()

  # global vars
  DATASET_NAME = args.dataset_name
  MODEL_NAME = args.model_name
  PERC_TEST = args.perc_test
  EPOCHS = args.epochs
  INIT_WINDOW = args.init_window
  M_WINDOW = args.m_window
  H_WINDOW = args.h_window
  END_WINDOW = H_WINDOW
  TEST_PREFIX_PERC = f"test_{str(PERC_TEST).replace('.',',')}"
  TEST_MODEL_ENTROPY = args.test_model_entropy
  EVALUATE_AUTO = args.test_model_entropy.startswith('auto')

  # -calculate_entropy
  if args.calculate_entropy:
    df = get_df_trajects()
    calc_trajects_entropy(df)
    dump_df_trajects(df)

  # -compare_results
  elif args.compare_results:
    compare_results()

  # -train or -evaluate
  elif args.train or args.evaluate:

    config.loginf('DATASET=' + (DATASET_NAME if DATASET_NAME != 'all' else repr(config.DS_NAMES)))
    dataset_suffix = '' if args.dataset_name == 'all' else f'_{DATASET_NAME}'
    model_ds_prefix = join(config.DATADIR, f'{MODEL_NAME}{dataset_suffix}')

    # -train: MODEL_FOLDER, MODEL_WEIGHTS
    if args.train:
      MODEL_FOLDER = model_ds_prefix + ('' if args.train_entropy == 'all' else
                                            f'_{args.train_entropy}_entropy')
      MODEL_WEIGHTS = join(MODEL_FOLDER, 'weights.hdf5')
      config.loginf(f'MODEL_FOLDER={MODEL_FOLDER}')
      config.loginf(f'MODEL_WEIGHTS={MODEL_WEIGHTS}')
      if not exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # -evaluate: MODEL_WEIGHTS
    if args.evaluate:
      MODEL_FOLDER = model_ds_prefix + ('' if args.test_model_entropy == 'all' else
                                            f'_{args.test_model_entropy}_entropy')
      if not exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
      config.loginf(f'MODEL_FOLDER={MODEL_FOLDER}')
      EVALUATE_NAME = f'{TEST_PREFIX_PERC}_{args.test_entropy}'
      config.loginf(f'EVALUATE_NAME at MODEL_FOLDER={EVALUATE_NAME}')

      # check existing with using one model
      if not args.test_model_entropy.startswith('auto'):
        MODEL_WEIGHTS = join(MODEL_FOLDER, 'weights.hdf5')
        assert exists(MODEL_WEIGHTS)
        config.loginf(f'MODEL_WEIGHTS={MODEL_WEIGHTS}')
      # check exists mutiple model when using auto select model
      if args.evaluate and args.test_model_entropy.startswith('auto'):
        MODEL_WEIGHTS_LOW = join(model_ds_prefix + "_low_entropy",
                                  'weights.hdf5')
        MODEL_WEIGHTS_MEDIUM = join(model_ds_prefix + "_medium_entropy",
                                    'weights.hdf5')
        MODEL_WEIGHTS_HIGHT = join(model_ds_prefix + "_hight_entropy",
                                    'weights.hdf5')
        assert exists(MODEL_WEIGHTS_LOW)
        assert exists(MODEL_WEIGHTS_MEDIUM)
        assert exists(MODEL_WEIGHTS_HIGHT)
        config.loginf('MODEL_WEIGHTS_LOW='+ MODEL_WEIGHTS_LOW)
        config.loginf('MODEL_WEIGHTS_MEDIUM='+ MODEL_WEIGHTS_MEDIUM)
        config.loginf('MODEL_WEIGHTS_HIGHT='+ MODEL_WEIGHTS_HIGHT)

    # partioning
    config.loginf('')
    config.loginf('partioning train/test ...')
    config.loginf(f'PERC_TEST is {PERC_TEST}')
    PARTITION_IDS = {}
    DF_TRAJECTS = get_df_trajects()
    if args.dataset_name != 'all':
      DF_TRAJECTS = DF_TRAJECTS[DF_TRAJECTS['ds'] == DATASET_NAME]

    # -train x_train, x_test
    if args.train:
      config.loginf(f'x_train, x_test entropy is {args.train_entropy}')
      x_train, x_test = get_train_test_split(DF_TRAJECTS, args.train_entropy, PERC_TEST)
    # -evaluate x_test, VIDEOS_TEST, USERS_TEST
    elif args.evaluate:
      config.loginf(f'x_test entropy={args.test_entropy}')
      _, x_test = get_train_test_split(DF_TRAJECTS, args.test_entropy, PERC_TEST)
      VIDEOS_TEST = x_test['ds_video'].unique()
      USERS_TEST = x_test['ds_user'].unique()

    # PARTITION_IDS
    if args.train:
      assert not x_train.empty
      fmt = 'x_train has {} trajectories: {} low, {} medium, {} hight'
      t_len = len(x_train)
      l_len = len(x_train[x_train['traject_entropy_class'] == 'low'])
      m_len = len(x_train[x_train['traject_entropy_class'] == 'medium'])
      h_len = len(x_train[x_train['traject_entropy_class'] == 'hight'])
      config.loginf(fmt.format(t_len, l_len, m_len, h_len))
      PARTITION_IDS['train'] = [
          {
              'video': row[1]['ds_video'],
              'user': row[1]['ds_user'],
              'trace_id': trace_id
          } for row in x_train.iterrows()
          for trace_id in range(INIT_WINDOW, row[1]['traject'].shape[0] - END_WINDOW)
      ]
      p_len = len(PARTITION_IDS['train'])
      config.loginf("PARTITION_IDS['train'] has {} positions".format(p_len))
    assert not x_test.empty
    fmt = 'x_test has {} trajectories: {} low, {} medium, {} hight'
    t_len = len(x_test)
    l_len = len(x_test[x_test['traject_entropy_class'] == 'low'])
    m_len = len(x_test[x_test['traject_entropy_class'] == 'medium'])
    h_len = len(x_test[x_test['traject_entropy_class'] == 'hight'])
    config.loginf(fmt.format(t_len, l_len, m_len, h_len))
    PARTITION_IDS['test'] = [
        {
            'video': row[1]['ds_video'],
            'user': row[1]['ds_user'],
            'trace_id': trace_id,
            'traject_entropy_class': row[1]['traject_entropy_class']
        } for row in x_test.iterrows()
        for trace_id in range(INIT_WINDOW, row[1]['traject'].shape[0] - END_WINDOW)
    ]
    p_len = len(PARTITION_IDS['test'])
    config.loginf("PARTITION_IDS['test'] has {} positions".format(p_len))

    # creating model
    config.loginf('')
    config.loginf('creating model ...')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if args.gpu_id:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    MODEL = create_model()
    assert MODEL

    # train, evaluate actions
    if args.train:
      config.loginf('')
      config.loginf('training ...')
      config.loginf(f'EPOCHS is {EPOCHS}')
      train()
    elif args.evaluate:
      config.loginf('')
      config.loginf('evaluating ...')
      evaluate()

  sys.exit()