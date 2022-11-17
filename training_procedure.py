#!env python

import argparse
import logging
import os
from contextlib import redirect_stderr
from os.path import basename, exists, isdir, join
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from users360 import *

logging.basicConfig(level=logging.INFO, format='-- %(filename)s: %(message)s')


# consts
METRIC = all_metrics['orthodromic']
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
RATE = 0.2
BATCH_SIZE = 128.0


def create_model() -> Any:
    if MODEL_NAME == "pos_only":
        from users360.head_motion_prediction.position_only_baseline import \
            create_pos_only_model
        return create_pos_only_model(M_WINDOW, H_WINDOW)
    else:
        raise NotImplemented()


def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch) -> np.array:
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2])
                         for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
    return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
    positions = positions * np.array([2 * np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)


def generate_arrays(list_IDs, future_window) -> Generator:
    while True:
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        count = 0
        np.random.shuffle(list_IDs)
        for IDs in list_IDs:
            user = IDs['user']
            video = IDs['video']
            x_i = IDs['time-stamp']
            # Load the data
            if MODEL_NAME == 'pos_only':
                encoder_pos_inputs_for_batch.append(get_traces(video, user, DATASET_NAME)[x_i - M_WINDOW:x_i])
                decoder_pos_inputs_for_batch.append(get_traces(video, user, DATASET_NAME)[x_i:x_i + 1])
                decoder_outputs_for_batch.append(get_traces(video, user, DATASET_NAME)[x_i + 1:x_i + future_window + 1])
            else:
                raise NotImplementedError()
            count += 1
            if count == BATCH_SIZE:
                count = 0
                if MODEL_NAME == 'pos_only':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
                else:
                    raise NotImplementedError()
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if MODEL_NAME == 'pos_only':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
            else:
                raise NotImplementedError()


def train() -> None:
    with redirect_stderr(open(os.devnull, "w")):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow.keras as keras

    steps_per_ep_train = np.ceil(len(PARTITION['train']) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(PARTITION['test']) / BATCH_SIZE)

    # train
    csv_logger_f = join(MODEL_FOLDER, 'train_results.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
    tb_callback = keras.callbacks.TensorBoard(log_dir=f'{MODEL_FOLDER}/logs')
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    if MODEL_NAME == 'pos_only':
        MODEL.fit_generator(
            generator=generate_arrays(PARTITION['train'], future_window=H_WINDOW),
            verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS,
            callbacks=[csv_logger, model_checkpoint, tb_callback],
            validation_data=generate_arrays(PARTITION['test'], future_window=H_WINDOW), validation_steps=steps_per_ep_validate
        )
    else:
        raise NotImplementedError()


def evaluate() -> None:
    if MODEL_NAME == "pos_only":
        MODEL.load_weights(MODEL_WEIGHTS)
    else:
        raise NotImplementedError()
    errors_per_video = {}
    errors_per_timestep = {}

    for ID in tqdm(PARTITION['test'], desc="position predictions"):
        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if video not in errors_per_video.keys():
            errors_per_video[video] = {}

        # Load the data
        if MODEL_NAME == 'pos_only':
            encoder_pos_inputs_for_sample = np.array([get_traces(video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]])
            decoder_pos_inputs_for_sample = np.array([get_traces(video, user, DATASET_NAME)[x_i:x_i + 1]])
        else:
            raise NotImplementedError()

        groundtruth = get_traces(video, user, DATASET_NAME)[x_i + 1:x_i + H_WINDOW + 1]

        if MODEL_NAME == 'pos_only':
            model_pred = MODEL.predict([transform_batches_cartesian_to_normalized_eulerian(
                encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
        else:
            raise NotImplementedError()

        for t in range(len(groundtruth)):
            if t not in errors_per_video[video].keys():
                errors_per_video[video][t] = []
            errors_per_video[video][t].append(METRIC(groundtruth[t], model_prediction[t]))
            if t not in errors_per_timestep.keys():
                errors_per_timestep[t] = []
            errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

    result_basefilename = join(MODEL_FOLDER, PERC_TEST_PREFIX)

    # avg_error_per_video
    avg_error_per_video = []
    for video_name in VIDEOS_TEST:
        for t in range(H_WINDOW):
            if not video_name in errors_per_video:
                logging.error(f'missing {video_name} in VIDEOS_TEST')
                continue
            avg = np.mean(errors_per_video[video_name][t])
            avg_error_per_video.append(f"video={video_name} {t} {avg}")
    result_file = f'{result_basefilename}_avg_error_per_video.csv'
    np.savetxt(result_file, avg_error_per_video, fmt='%s')
    logging.info(f"saving {result_file}")

    # avg_error_per_timestep
    avg_error_per_timestep = []
    for t in range(H_WINDOW):
        avg = np.mean(errors_per_timestep[t])
        avg_error_per_timestep.append(avg)
    # avg_error_per_timestep.csv
    result_file = f"{result_basefilename}_avg_error_per_timestep"
    logging.info(f"saving {result_file}.csv")
    np.savetxt(f'{result_file}.csv', avg_error_per_timestep)

    # avg_error_per_timestep.png
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlim(2.5)
    plt.xlabel('Prediction step s (sec.)')
    logging.info(f"saving {result_file}.png")
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    plt.savefig(result_file, bbox_inches='tight')


def compare_results() -> None:
    suffix = '_avg_error_per_timestep.csv'
    prefix = PERC_TEST_PREFIX

    # find files with suffix
    dirs = [d for d in os.listdir(DATADIR) if d.startswith(MODEL_NAME)]
    csv_file_l = [(dir, f) for dir in dirs for f in os.listdir(join(DATADIR, dir))
                  if (f.endswith(suffix) and f.startswith(prefix))]
    csv_data_l = [(dir, f, np.loadtxt(join(DATADIR, dir, f))) for (dir, f) in csv_file_l]
    assert csv_data_l, f"you should first run {basename(__file__)} -evaluate -entropy <all,low,medium,hight>"

    # sort by the last horizon hight
    last_horizon_avg = lambda item: item[2][-1]  # [2] is csv_data [-1] is last horizon
    csv_data_l.sort(reverse=True, key=last_horizon_avg)

    # plot as image
    add_plot = lambda dir, file, csv_data: plt.plot(np.arange(H_WINDOW) + 1 * RATE, csv_data, label=f"{dir}_{file.removesuffix(suffix)}")
    [add_plot(*csv_data) for csv_data in csv_data_l]
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlim(2.5)
    plt.xlabel('Prediction step s (sec.)')
    result_file = join(DATADIR, f"compare_{MODEL_NAME}")
    logging.info(f"saving {result_file}.png")
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    plt.savefig(result_file, bbox_inches='tight')


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.description = 'train or evaluate users360 models and datasets'
    model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
    entropy_classes = ['all', 'low', 'medium', 'hight']
    dataset_names = ['all', 'David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']

    # main actions params
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-load_raw_dataset', action='store_true', help='Load raw dataset and save it as pickle ')
    group.add_argument('-calculate_entropy', action='store_true', help='Calculate trajectories entropy')
    group.add_argument('-compare_results', action='store_true', help='Show a comparison of -evaluate results')
    group.add_argument('-train', action='store_true', help='Train model')
    group.add_argument('-evaluate', action='store_true', help='Evaluate model')

    # train only params
    parser.add_argument('-epochs', nargs='?', type=int, default=500,
                        help='epochs numbers (default is 500)')
    parser.add_argument('-train_entropy', nargs='?', default=entropy_classes[0],
                        choices=entropy_classes, help='Name of entropy_class to filter training/evalaute')

    # evaluate only params
    parser.add_argument('-test_entropy', nargs='?', default='same',
                        choices=entropy_classes, help='Name of entropy_class to filter training/evalaute')

    # train/evaluate params
    parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
                        help='Used cuda gpu (default: 0)')
    parser.add_argument('-model_name', nargs='?', choices=model_names, default=model_names[0],
                        help='The name of the reference model to used (default: pos_only)')
    parser.add_argument('-dataset_name', nargs='?', choices=dataset_names, default=dataset_names[0],
                        help='The name of the dataset used to train this network  (default: all)')
    parser.add_argument('-init_window', nargs='?', type=int, default=30,
                        help='Initial buffer to avoid stationary part (default: 30)')
    parser.add_argument('-m_window', nargs='?', type=int, default=5,
                        help='Buffer window in timesteps (default: 5)')
    parser.add_argument('-h_window', nargs='?', type=int, default=25,
                        help='Forecast window in timesteps (5 timesteps = 1 second) used to predict (default: 25)')
    parser.add_argument('-perc_test', nargs='?', type=float, default=0.2,
                        help='Test percetage (default: 0.2)')

    args = parser.parse_args()
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    DATASET_NAME = args.dataset_name
    INIT_WINDOW = args.init_window
    M_WINDOW = args.m_window
    H_WINDOW = args.h_window
    END_WINDOW = H_WINDOW
    PERC_TEST = args.perc_test

    # used in multiple actions
    PERC_TEST_PREFIX = f"test{str(PERC_TEST).replace('.',',')}"
    model_entropy_sufix = f'_{args.train_entropy}_entropy' if args.train_entropy != 'all' else ''
    MODEL_FOLDER = join(DATADIR, f'{MODEL_NAME}_{DATASET_NAME}{model_entropy_sufix}')
    if not exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    logging.info(f"MODEL_FOLDER is {MODEL_FOLDER}")

    # not -train, -evaluation actions
    if args.load_raw_dataset:
        Data.instance().save()
        exit()
    if args.calculate_entropy:
        calc_trajects_entropy()
        Data.instance().save()
        exit()
    if args.compare_results:
        compare_results()
        exit()

    # -train, -evaluation actions
    # partioning
    logging.info("")
    logging.info("partioning train/test ...")
    df = get_df_trajects()
    args.test_entropy = args.train_entropy if args.test_entropy == "same" else args.test_entropy
    logging.info(f"X_train entropy is {args.train_entropy}")
    logging.info(f"X_test entropy is {args.test_entropy}")
    X_train, X_test = [], []
    if (args.train_entropy != 'all') and (args.test_entropy != 'all'):
        assert not df['entropy_class'].empty, "dataset has no 'entropy_class' collumn, run -calc_trajects_entropy"
        X_train, X_test = train_test_split(df, test_size=PERC_TEST, random_state=1)
    else:
        if args.train_entropy == 'all':
            X_train, _ = train_test_split(df, test_size=PERC_TEST, random_state=1)
        else:
            X_train, _ = train_test_split(df[df['entropy_class'] == args.train_entropy], test_size=PERC_TEST, random_state=1)
        if args.test_entropy == 'all':
            _, X_test = train_test_split(df, test_size=PERC_TEST, random_state=1)
        else:
            _, X_test = train_test_split(df[df['entropy_class'] == args.test_entropy], test_size=PERC_TEST, random_state=1)
    assert (not X_test.empty and not X_train.empty)
    logging.info(f"PERC_TRAIN is {1-PERC_TEST}")
    logging.info(f"X_train has {len(X_train)} trajectories")
    logging.info(f"X_test has {len(X_test)} trajectories")
    PARTITION = {}
    PARTITION['train'] = [{'video': row[1]['ds_video'], 'user': row[1]
                          ['ds_user'], 'time-stamp': tstap}
                          for row in X_train.iterrows()
                          for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
    PARTITION['test'] = [{'video': row[1]['ds_video'], 'user': row[1]
                         ['ds_user'], 'time-stamp': tstap}
                         for row in X_test.iterrows()
                         for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
    VIDEOS_TEST = X_test['ds_video'].unique()
    USERS_TEST = X_test['ds_user'].unique()
    logging.info(f"PARTITION['train'] has {len(PARTITION['train'])} position predictions")
    logging.info(f"PARTITION['test'] has {len(PARTITION['test'])} position predictions")

    # creating model
    logging.info("")
    logging.info("creating model ...")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    MODEL_WEIGHTS = join(MODEL_FOLDER, 'weights.hdf5')
    logging.info(f"MODEL_WEIGHTS={MODEL_WEIGHTS}")
    MODEL = create_model()

    if args.train:
        logging.info("")
        logging.info("training ...")
        logging.info(f"EPOCHS is {EPOCHS}")
        train()
    elif args.evaluate:
        logging.info("")
        logging.info("evaluating ...")
        assert exists(MODEL_WEIGHTS), f"{MODEL_WEIGHTS} does not exists"
        logging.info(f"evaluate_entropy is {args.train_entropy}")
        evaluate()
