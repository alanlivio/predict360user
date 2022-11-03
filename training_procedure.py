#!env python

import argparse
import logging
import os
from contextlib import redirect_stderr
from os.path import exists, isdir, join
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np

with redirect_stderr(open(os.devnull, "w")):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import keras
    import tensorflow.keras as keras
    import keras_tqdm

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


def create_model() -> keras.models.Model:
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

    result_basefilename = join(MODEL_FOLDER, f"test{str(PERC_TEST).replace('.',',')}")

    # avg_error_per_video
    avg_error_per_video = []
    for video_name in VIDEOS_TEST:
        for t in range(H_WINDOW):
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
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlim(2.5)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    result_file = f"{result_basefilename}_avg_error_per_timestep"
    logging.info(f"saving {result_file}.csv")
    np.savetxt(f'{result_file}.csv', avg_error_per_timestep)
    logging.info(f"saving {result_file}.png")
    plt.savefig(result_file)


def compare_results() -> None:
    dirs = [d for d in os.listdir(DATADIR) if d.startswith(MODEL_NAME)]
    for dir in dirs:
        filename = next(f for f in os.listdir(join(DATADIR, dir)) if f.endswith('_avg_error_per_timestep.csv'))
        csv = np.loadtxt(join(DATADIR, dir, filename))
        plt.plot(np.arange(H_WINDOW) + 1 * RATE, csv, label=dir)
    assert dirs, 'none results'
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlim(2.5)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    result_file = join(DATADIR, f"compare_{MODEL_NAME}")
    logging.info(f"saving {result_file}.png")
    plt.savefig(result_file)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.description = 'train or evaluate users360 models and datasets'
    model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
    entropy_classes = ['all', 'low', 'medium', 'hight']
    dataset_names = ['all', 'David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']

    # main actions params
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-train', action='store_true',
                       help='Flag that tells run the train procedure')
    group.add_argument('-evaluate', action='store_true',
                       help='Flag that tells run the evaluate procedure')
    group.add_argument('-compare_results', action='store_true',
                       help='Flag that tells run the train procedure')
    group.add_argument('-calculate_entropy', action='store_true',
                       help='Flag that tells run the calculate entropy procedure')

    # train only params
    parser.add_argument('-epochs', nargs='?', type=int, default=500,
                        help='epochs numbers (default is 500)')

    # train/evaluate params
    parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
                        help='Used cuda gpu')
    parser.add_argument('-model_name', nargs='?',
                        choices=model_names, default=model_names[0],
                        help='The name of the model used to reference the network structure used')
    parser.add_argument('-dataset_name', nargs='?',
                        choices=dataset_names, default=dataset_names[0],
                        help='The name of the dataset used to train this network')
    parser.add_argument('-init_window', nargs='?', type=int, default=30,
                        help='Initial buffer to avoid stationary part')
    parser.add_argument('-m_window', nargs='?', type=int, default=5,
                        help='Buffer window in timesteps',)
    parser.add_argument('-h_window', nargs='?', type=int, default=25,
                        help='Forecast window in timesteps used to predict (5 timesteps = 1 second)')
    parser.add_argument('-perc_test', nargs='?', type=float, default=0.2,
                        help='Test percetage (default is 0.2)')
    parser.add_argument('-entropy', nargs='?', default=entropy_classes[0],
                        choices=entropy_classes, help='Name of entropy_class to filter training/evalaute')
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    DATASET_NAME = args.dataset_name
    INIT_WINDOW = args.init_window
    M_WINDOW = args.m_window
    H_WINDOW = args.h_window
    END_WINDOW = H_WINDOW
    PERC_TEST = args.perc_test

    # prepare variables/partitions/model for train/evaluate
    if (args.calculate_entropy):
        calc_trajects_entropy()
        Data.instance.save()
    if (args.train or args.evaluate):
        # -- variables
        # DATASET_SAMPLED_FOLDER
        # DATASET_DIR_HMP = join('users360', 'head_motion_prediction', DATASET_NAME)
        # DATASET_SAMPLED_FOLDER = join(DATASET_DIR_HMP, 'sampled_dataset')
        # MODEL_FOLDER
        # WIN_PARAMS = 'init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + \
        # str(H_WINDOW) + '_end_' + str(END_WINDOW)
        ENTROPY_SUFIX = f'_{args.entropy}_entropy' if args.entropy != 'all' else ''
        if MODEL_NAME == 'pos_only':
            MODEL_FOLDER = join(DATADIR, f'{MODEL_NAME}_{DATASET_NAME}{ENTROPY_SUFIX}')
        else:
            raise NotImplementedError()
        MODEL_WEIGHTS = join(MODEL_FOLDER, 'weights.hdf5')
        if not exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        logging.info(f"MODEL_FOLDER is {MODEL_FOLDER}")
        # if not train, check if MODEL_WEIGHTS exists

        # -- partitions
        logging.info("")
        logging.info("preparing train/test partitions ...")
        df = get_df_trajects()
        if args.entropy == 'all':
            X, Y = train_test_split(df, test_size=PERC_TEST, random_state=1)
        else:
            if not 'entropy_class' in df.columns:
                calc_trajects_entropy()
            assert not df['entropy_class'].empty, f"df has no 'entropy_class' collumn "
            X, Y = train_test_split(df[df['entropy_class'] == args.entropy], test_size=PERC_TEST, random_state=1)
            assert not X.empty, f"{DATASET_NAME} train partition has none traject with {args.entropy} entropy "
            assert not Y.empty, f"{DATASET_NAME} test partition has none traject with {args.entropy} entropy "
        PARTITION = {}
        PARTITION['train'] = [{'video': row[1]['ds_video'], 'user': row[1]
                               ['ds_user'], 'time-stamp': tstap}
                              for row in X.iterrows()
                              for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
        PARTITION['test'] = [{'video': row[1]['ds_video'], 'user': row[1]
                              ['ds_user'], 'time-stamp': tstap}
                             for row in Y.iterrows()
                             for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
        VIDEOS_TEST = Y['ds_video'].unique()
        USERS_TEST = Y['ds_user'].unique()

        # -- model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logging.info("")
        logging.info("creating model ...")
        MODEL = create_model()

    if args.train:
        logging.info("")
        logging.info("training ...")
        logging.info(f"EPOCHS is {EPOCHS}")
        logging.info(f"PERC_TRAIN is {1-PERC_TEST}")
        logging.info(f"train_entropy is {args.entropy}")
        logging.info(f"X has {len(X)} trajects")
        logging.info(f"PARTITION['train'] has {len(PARTITION['train'])} position predictions")
        train()
    elif args.evaluate:
        logging.info("")
        logging.info("evaluating ...")
        assert exists(MODEL_WEIGHTS), f"{MODEL_WEIGHTS} does not exists"
        logging.info(f"PERC_TRAIN is {PERC_TEST}")
        logging.info(f"evaluate_entropy is {args.entropy}")
        logging.info(f"Y has {len(Y)} trajects")
        logging.info(f"PARTITION['test'] has {len(PARTITION['test'])} position predictions")
        evaluate()
    elif args.compare_results:
        logging.info("")
        logging.info("comparing results ...")
        compare_results()
