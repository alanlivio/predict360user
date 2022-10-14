#!env python
import argparse
import logging
import sys

import numpy as np
import tensorflow.keras as keras

from users360 import *

# consts
METRIC = all_metrics['orthodromic']
EPOCHS = 500
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
RATE = 0.2
# PERC_VIDEOS_TRAIN = 0.8
# PERC_USERS_TRAIN = 0.5
PERC_VIDEOS_TRAIN = 0.99
PERC_USERS_TRAIN = 0.99
BATCH_SIZE = 128.0

# vars from argparse
ARGS = None
DATASET_NAME: str
MAKE_DATASET = False
TRAIN_MODEL = False
EVALUATE_MODEL = False
MODEL_NAME: str
M_WINDOW: int
H_WINDOW: int
INIT_WINDOW: int
END_WINDOW: int

# other vars
MODEL = None
USERS = None
VIDEOS = None
VIDEOS_TRAIN = None 
VIDEOS_TEST = None
USERS_TRAIN = None
USERS_TEST = None
USERS_PER_VIDEO = None
PARTITION = None
ALL_TRACES = None
RESULTS_FOLDER: str
MODELS_FOLDER: str
DATASET_SAMPLED_FOLDER: str
EXP_NAME: str

from users360.head_motion_prediction.position_only_baseline import \
    create_pos_only_model


def create_model(name=""):
    return create_pos_only_model(M_WINDOW, H_WINDOW)

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2])
                         for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
    return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2 * np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)


# def get_traces(video, user):
#     row = Data.singleton().df_trajects.query(f"ds={DATASET_NAME} and ds_user={user} and ds_video={video}")
#     return row['traces']

# get videos
def get_video_ids():
    # Returns the ids of the videos in the dataset
    list_of_videos = [o for o in os.listdir(DATASET_SAMPLED_FOLDER) if not o.endswith('.gitkeep')]
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_videos.sort()
    return list_of_videos

def get_user_ids():
    # returns the unique ids of the users in the dataset
    videos = get_video_ids()
    users = []
    for video in videos:
        for user in [o for o in os.listdir(os.path.join(DATASET_SAMPLED_FOLDER, video)) if not o.endswith('.gitkeep')]:
            users.append(user)
    list_of_users = list(set(users))
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_users.sort()
    return list_of_users

def get_users_per_video():
    # Returns a dictionary indexed by video, and under each index you can find the users for which the trace has been stored for this video
    videos = get_video_ids()
    users_per_video = {}
    for video in videos:
        users_per_video[video] = [user for user in os.listdir(os.path.join(DATASET_SAMPLED_FOLDER, video))]
    return users_per_video
    
def read_sampled_positions_for_trace(video, user):
    # returns only the positions from the trace
    # ~time-stamp~ is removed from the output, only x, y, z (in 3d coordinates) is returned
    path = os.path.join(DATASET_SAMPLED_FOLDER, video, user)
    data = pd.read_csv(path, header=None)
    return data.values[:, 1:]

def read_sampled_data_for_trace(video, user):
    # returns the whole data organized as follows:
    # time-stamp, x, y, z (in 3d coordinates)
    path = os.path.join(DATASET_SAMPLED_FOLDER, video, user)
    data = pd.read_csv(path, header=None)
    return data.values

def split_list_by_percentage(the_list, percentage):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Shuffle to select randomly
    np.random.shuffle(the_list)
    num_samples_first_part = int(len(the_list) * percentage)
    train_part = the_list[:num_samples_first_part]
    test_part = the_list[num_samples_first_part:]
    return train_part, test_part

def generate_arrays(list_IDs, future_window):
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
                encoder_pos_inputs_for_batch.append(ALL_TRACES[video][user][x_i-M_WINDOW:x_i])
                decoder_pos_inputs_for_batch.append(ALL_TRACES[video][user][x_i:x_i+1])
                decoder_outputs_for_batch.append(ALL_TRACES[video][user][x_i+1:x_i+future_window+1])
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

def train():
    steps_per_ep_train = np.ceil(len(PARTITION['train']) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(PARTITION['test']) / BATCH_SIZE)
    
    # train
    csv_logger_f = os.path.join(RESULTS_FOLDER, 'results.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
    weights_f = os.path.join(MODELS_FOLDER, 'weights.hdf5')
    model_checkpoint = keras.callbacks.ModelCheckpoint(weights_f, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    if MODEL_NAME == 'pos_only':
        MODEL.fit_generator(
            generator=generate_arrays(PARTITION['train'], future_window=H_WINDOW),
            verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS,
            callbacks=[csv_logger, model_checkpoint],
            validation_data=generate_arrays(PARTITION['test'], future_window=H_WINDOW), validation_steps=steps_per_ep_validate
        )
    else:
        raise NotImplementedError()

def evaluate():
    
    if MODEL_NAME == "pos_only":
        MODEL.load_weights(MODELS_FOLDER + '/weights.hdf5')
    else:
        raise NotImplementedError()
    
    traces_count = 0
    errors_per_video = {}
    errors_per_timestep = {}

    for ID in PARTITION['test']:
        traces_count += 1
        logging.info(f"Progress: {traces_count}/{len(PARTITION['test'])}")

        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if video not in errors_per_video.keys():
            errors_per_video[video] = {}

        # Load the data
        if MODEL_NAME == 'pos_only':
            encoder_pos_inputs_for_sample = np.array([ALL_TRACES[video][user][x_i-M_WINDOW:x_i]])
            decoder_pos_inputs_for_sample = np.array([ALL_TRACES[video][user][x_i:x_i + 1]])
        else:
            raise NotImplementedError()
        
        groundtruth = ALL_TRACES[video][user][x_i+1:x_i+H_WINDOW+1]

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

    for video_name in VIDEOS_TEST:
        for t in range(H_WINDOW):
            logging.info(f"video={video_name} {t} {np.mean(errors_per_video[video_name][t])}")

    import matplotlib.pyplot as plt
    avg_error_per_timestep = []
    for t in range(H_WINDOW):
        logging.info(f"Average {t} {np.mean(errors_per_timestep[t])}")
        avg_error_per_timestep.append(np.mean(errors_per_timestep[t]))
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep, label=MODEL_NAME)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    plt.show()


def make_dataset():
    Data.singleton().load_dataset()
    df = Data.singleton().df_trajects
    logging.info(f"df_trajects.size={df.size}")
    logging.info("calc_trajects_entropy")
    calc_trajects_entropy()
    Data.singleton().save()


if __name__ == "__main__":
    if not exists("data/Data.pickle"):
        logging.error("data/Data.pickle does not exist. Run python main.py --make_dataset")
        exit

    # create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.description = 'train or evaluate users360 models and datasets'
    model_names = ['pos_only']
    # model_names = ['TRACK', 'CVPR18', 'pos_only', 'no_motion', 'most_salient_point', 'true_saliency',
    #                'content_based_saliency', 'CVPR18_orig', 'TRACK_AblatSal', 'TRACK_AblatFuse', 'MM18', 'pos_only_3d_loss']
    dataset_names = ['David_MMSys_18']
    parser.add_argument('-make_dataset', action='store_true',
                        help='Flag that tells run make_dataset procedure')
    parser.add_argument('-train', action='store_true',
                        help='Flag that tells run the train procedure')
    parser.add_argument('-evaluate', action='store_true',
                        help='Flag that tells run the evaluate procedure')
    parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
                        help='Used cuda gpu')
    parser.add_argument('-model_name', nargs='?',
                        choices=model_names, type=str, default=model_names[0],
                        help='The name of the model used to reference the network structure used')
    parser.add_argument('-dataset_name', nargs='?',
                        choices=dataset_names, type=str, default=dataset_names[0],
                        help='The name of the dataset used to train this network')
    parser.add_argument('-init_window', nargs='?', type=int, default=30,
                        help='Initial buffer to avoid stationary part')
    parser.add_argument('-m_window', nargs='?', type=int, default=5,
                        help='Buffer window in timesteps',)
    parser.add_argument('-h_window', nargs='?', type=int, default=25,
                        help='Forecast window in timesteps used to predict (5 timesteps = 1 second)')

    # vars from argparse
    ARGS = parser.parse_args()
    TRAIN_MODEL = ARGS.train
    EVALUATE_MODEL = ARGS.evaluate
    MODEL_NAME = ARGS.model_name
    DATASET_NAME = ARGS.dataset_name
    INIT_WINDOW = ARGS.init_window
    M_WINDOW = ARGS.m_window
    H_WINDOW = ARGS.h_window
    END_WINDOW = H_WINDOW
    
    # DATASET_SAMPLED_FOLDER  
    DATASET_DIR_HMP = os.path.join('users360','head_motion_prediction', DATASET_NAME)
    DATASET_SAMPLED_FOLDER = os.path.join(DATASET_DIR_HMP, 'sampled_dataset')
    
    # RESULTS_FOLDER, MODELS_FOLDER folders
    DATASET_DIR = os.path.join(DATADIR, DATASET_NAME)
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    if MODEL_NAME == 'pos_only':
        RESULTS_FOLDER = os.path.join(DATASET_DIR, 'pos_only',  'Results_EncDec_eulerian' + EXP_NAME)
        MODELS_FOLDER = os.path.join(DATASET_DIR, 'pos_only' , 'Models_EncDec_eulerian' + EXP_NAME)
    else:
        raise NotImplementedError()
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    # prepare partitions/model for train/evaluate
    if (ARGS.train or ARGS.evaluate): 
        logging.info("prepare partitions")
        USERS = get_user_ids()
        VIDEOS = get_video_ids()
        USERS_PER_VIDEO = get_users_per_video()
        ALL_TRACES = {}
        for video in VIDEOS:
            ALL_TRACES[video] = {}
            for user in USERS_PER_VIDEO[video]:
                ALL_TRACES[video][user] = read_sampled_positions_for_trace(str(video), str(user))
        
        # split
        VIDEOS_TRAIN, VIDEOS_TEST = split_list_by_percentage(VIDEOS, PERC_VIDEOS_TRAIN)
        USERS_TRAIN, USERS_TEST = split_list_by_percentage(USERS, PERC_USERS_TRAIN)

        PARTITION = {}
        PARTITION['train'] = []
        PARTITION['test'] = []
        for video in VIDEOS_TRAIN:
            for user in USERS_TRAIN:
                # to get the length of the trace
                trace_length = read_sampled_data_for_trace(video, user).shape[0]
                for tstap in range(INIT_WINDOW, trace_length - END_WINDOW):
                    ID = {'video': video, 'user': user, 'time-stamp': tstap}
                    PARTITION['train'].append(ID)
        for video in VIDEOS_TEST:
            for user in USERS_TEST:
                # to get the length of the trace
                trace_length = read_sampled_data_for_trace(video, user).shape[0]
                for tstap in range(INIT_WINDOW, trace_length - END_WINDOW):
                    ID = {'video': video, 'user': user, 'time-stamp': tstap}
                    PARTITION['test'].append(ID)
        # create model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if ARGS.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
        logging.info("create model")
        MODEL = create_model()
    if ARGS.make_dataset:
        logging.info("make_dataset")
        make_dataset()
    if ARGS.train:
        logging.info("train")
        train()
    if ARGS.evaluate:
        logging.info("evaluate")
        evaluate()
