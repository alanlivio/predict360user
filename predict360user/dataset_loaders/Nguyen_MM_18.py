import os
import pickle
from os import makedirs
from os.path import exists, join

import cv2
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from predict360user.data_ingestion import DATADIR
from predict360user.utils.math360 import *

ROOT_FOLDER = join(DATADIR, "Nguyen_MM_18/dataset/")
OUTPUT_FOLDER = join(DATADIR, "Nguyen_MM_18/sampled_dataset")
OUTPUT_SALIENCY_FOLDER = join(DATADIR, "Nguyen_MM_18/extract_saliency/saliency")
OUTPUT_TRUE_SALIENCY_FOLDER = join(DATADIR, "Nguyen_MM_18/true_saliency")
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256
ORIGINAL_SAMPLING_RATE = 0.063
SAMPLING_RATE = 0.2
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216


# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
# CALCULATE DEGREE DISTANCE BETWEEN TWO 3D VECTORS
def unit_vector(vector) -> np.ndarray:
    return vector / np.linalg.norm(vector)


# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def degree_distance(v1, v2) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi * 180


# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def vector_to_ang(_v) -> tuple:
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])  # degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [
        0,
        np.cos(alpha / 180.0 * np.pi),
        0,
    ]  # proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = (
        _v - proj1
    )  # proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(
        proj2, [1, 0, 0]
    )  # theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi


def ang_to_geoxy(_theta, _phi, _h, _w) -> tuple:
    x = _h / 2.0 - (_h / 2.0) * np.sin(_phi / 180.0 * np.pi)
    temp = _theta
    if temp < 0:
        temp = 180 + temp + 180
    temp = 360 - temp
    y = temp * 1.0 / 360 * _w
    return int(x), int(y)


H = 10
W = 20


def create_fixation_map(v) -> np.ndarray:
    theta, phi = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H - hi - 1, W - wi - 1] = 1
    return result


def load_dataset():
    if "salient_ds_dict" not in locals():
        with open(join(ROOT_FOLDER, "salient_ds_dict_w16_h9"), "rb") as file_in:
            u = pickle._Unpickler(file_in)
            u.encoding = "latin1"
            salient_ds_dict = u.load()
    return salient_ds_dict


# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
# In equirectangular projection, the longitude ranges from left to right as follows: 0 +90 +180 -180 -90 0
# and the latitude ranges from top to bottom: -90 90
def get_original_dataset() -> dict:
    dataset = {}
    # load data from pickle file.
    salient_ds_dict = load_dataset()
    for video_id, video in enumerate(salient_ds_dict["360net"].keys()):
        time_stamps = salient_ds_dict["360net"][video]["timestamp"]
        id_sort_tstamps = np.argsort(time_stamps)
        for user_id in range(len(salient_ds_dict["360net"][video]["headpos"])):
            print(
                "get head positions from original dataset",
                "video",
                video_id,
                "/",
                len(salient_ds_dict["360net"].keys()),
                "user",
                user_id,
                "/",
                len(salient_ds_dict["360net"][video]["headpos"]),
            )
            user = str(user_id)
            if user not in dataset.keys():
                dataset[user] = {}
            positions_vector = salient_ds_dict["360net"][video]["headpos"][user_id]
            samples = []
            # Sorted time-stamps
            for id_sort in id_sort_tstamps:
                yaw_true, pitch_true = vector_to_ang(positions_vector[id_sort])
                samples.append(
                    {"sec": time_stamps[id_sort], "yaw": yaw_true, "pitch": pitch_true}
                )
            dataset[user][video] = samples
    return dataset


# In equirectangular projection, the longitude ranges from left to right as follows: 0 +90 +180 -180 -90 0
# and the latitude ranges from top to bottom: -90 90
### We will transform these coordinates so that
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates (after applying eulerian_to_cartesian function)
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
# For this, we will transform:
# pitch = -90 into pitch = 0
# pitch = 90 into pitch = pi
# yaw = -180 into yaw = 0
# yaw = -90 into yaw = pi/2
# yaw = 0 into yaw = pi
# yaw = 90 into yaw = 3pi/2
# yaw = 180 into yaw = 2pi
def transform_the_degrees_in_range(yaw, pitch) -> tuple:
    yaw = ((yaw + 180.0) / 360.0) * 2 * np.pi
    pitch = ((pitch + 90.0) / 180.0) * np.pi
    return yaw, pitch


# Performs the opposite transformation than transform_the_degrees_in_range
def transform_the_radians_to_original(yaw, pitch) -> tuple:
    yaw = ((yaw / (2 * np.pi)) * 360.0) - 180.0
    pitch = ((pitch / np.pi) * 180.0) - 90.0
    return yaw, pitch


# ToDo Copied exactly from David_MMSys_18/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate) -> dict:
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            sample_orig = np.array([1, 0, 0])
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(
                    sample["yaw"], sample["pitch"]
                )
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                quat_rot = rotationBetweenVectors(sample_orig, sample_new)
                # append the quaternion to the list
                data_per_video.append(
                    [sample["sec"], quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3]]
                )
                # update the values of time and sample
            # interpolate the quaternions to have a rate of 0.2 secs
            data_per_video = np.array(data_per_video)
            # In this case the time starts counting at random parts of the video
            dataset[user][video] = interpolate_quaternions(
                data_per_video[:, 0],
                data_per_video[:, 1:],
                rate=rate,
                time_orig_at_zero=False,
            )
    return dataset


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace) -> np.ndarray:
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        restored_yaw, restored_pitch = cartesian_to_eulerian(
            sample_new[0], sample_new[1], sample_new[2]
        )
        restored_yaw, restored_pitch = transform_the_radians_to_original(
            restored_yaw, restored_pitch
        )
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)


def recover_original_angles_from_xyz_trace(xyz_trace) -> np.ndarray:
    angles_per_video = []
    for sample in xyz_trace:
        restored_yaw, restored_pitch = cartesian_to_eulerian(
            sample[1], sample[2], sample[3]
        )
        restored_yaw, restored_pitch = transform_the_radians_to_original(
            restored_yaw, restored_pitch
        )
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)


# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace) -> np.ndarray:
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        angles_per_video.append(sample_new)
    return np.concatenate(
        (quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1
    )


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def get_xyz_dataset(sampled_dataset) -> dict:
    dataset = {}
    for user in sampled_dataset.keys():
        dataset[user] = {}
        for video in sampled_dataset[user].keys():
            dataset[user][video] = recover_xyz_from_quaternions_trace(
                sampled_dataset[user][video]
            )
    return dataset


# Store the dataset in xyz coordinates form into the folder_to_store
def store_dataset(xyz_dataset, folder_to_store) -> None:
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            video_folder = join(folder_to_store, video)
            # Create the folder for the video if it doesn't exist
            if not exists(video_folder):
                makedirs(video_folder)
            path = join(video_folder, user)
            df = pd.DataFrame(xyz_dataset[user][video])
            df.to_csv(path, header=False, index=False)


def create_and_store_sampled_dataset() -> None:
    original_dataset = get_original_dataset()
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)


# ToDo Copied exactly from Extract_Saliency/panosalnet
def post_filter(_img) -> np.ndarray:
    result = np.copy(_img)
    result[:3, :3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result


def create_saliency_maps() -> None:
    salient_ds_dict = load_dataset()
    for video in salient_ds_dict["360net"].keys():
        sal_per_vid = {}
        video_sal_folder = join(OUTPUT_SALIENCY_FOLDER, video)
        if not exists(video_sal_folder):
            makedirs(video_sal_folder)

        time_stamps = salient_ds_dict["360net"][video]["timestamp"]
        id_sort_tstamps = np.argsort(time_stamps)

        time_stamps_by_rate = np.arange(
            time_stamps[id_sort_tstamps[0]],
            time_stamps[id_sort_tstamps[-1]] + SAMPLING_RATE / 2.0,
            SAMPLING_RATE,
        )

        for tstap_id, sampled_timestamp in enumerate(time_stamps_by_rate):
            # get the saliency with closest time-stamp
            sal_id = np.argmin(np.power(time_stamps - sampled_timestamp, 2.0))
            saliency = salient_ds_dict["360net"][video]["salient"][sal_id]
            salient = cv2.resize(saliency, (NUM_TILES_WIDTH, NUM_TILES_HEIGHT))
            salient = salient * 1.0 - salient.min()
            salient = (salient / salient.max()) * 255
            salient = post_filter(salient)
            frame_id = "%03d" % (tstap_id + 1)
            sal_per_vid[frame_id] = salient
            output_file = join(video_sal_folder, frame_id + ".jpg")
            cv2.imwrite(output_file, salient)
            print("saved image %s" % (output_file))

        pickle.dump(sal_per_vid, open(join(video_sal_folder, video), "wb"))


# Returns the maximum number of samples among all users (the length of the largest trace)
def get_max_num_samples_for_video(video, sampled_dataset, users_in_video) -> int:
    max_len = 0
    for user in users_in_video:
        curr_len = len(sampled_dataset[user][video])
        if curr_len > max_len:
            max_len = curr_len
    return max_len


def create_and_store_true_saliency(sampled_dataset) -> None:
    if not exists(OUTPUT_TRUE_SALIENCY_FOLDER):
        makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)

    # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
    # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
    def from_position_to_tile_probability_cartesian(pos) -> np.ndarray:
        yaw_grid, pitch_grid = np.meshgrid(
            np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
            np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False),
        )
        yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
        pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
        yaw_grid = yaw_grid * 2 * np.pi
        pitch_grid = pitch_grid * np.pi
        x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
        great_circle_distance = np.arccos(
            np.maximum(
                np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0),
                -1.0,
            )
        )
        gaussian_orth = np.exp(
            (-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance)
        )
        return gaussian_orth

    videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith(".gitkeep")]
    users_per_video = {}
    for video in videos:
        users_per_video[video] = [
            user for user in os.listdir(os.path.join(OUTPUT_FOLDER, video))
        ]

    for enum_video, video in enumerate(videos):
        print(
            "creating true saliency for video", video, "-", enum_video, "/", len(videos)
        )
        real_saliency_for_video = []
        max_num_samples = get_max_num_samples_for_video(
            video, sampled_dataset, users_per_video[video]
        )
        for x_i in range(max_num_samples):
            tileprobs_for_video_cartesian = []
            for user in users_per_video[video]:
                if len(sampled_dataset[user][video]) > x_i:
                    tileprobs_cartesian = from_position_to_tile_probability_cartesian(
                        sampled_dataset[user][video][x_i, 1:]
                    )
                    tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = (
                np.sum(tileprobs_for_video_cartesian, axis=0)
                / tileprobs_for_video_cartesian.shape[0]
            )
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)
        true_sal_out_file = join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)


def load_sampled_dataset() -> dict:
    list_of_videos = [
        o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith(".gitkeep")
    ]
    dataset = {}
    for video in list_of_videos:
        for user in [
            o
            for o in os.listdir(join(OUTPUT_FOLDER, video))
            if not o.endswith(".gitkeep")
        ]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset
