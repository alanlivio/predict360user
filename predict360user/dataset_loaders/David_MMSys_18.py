import os
from os import makedirs
from os.path import exists, join

import cv2
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from predict360user import config
from predict360user.utils import (cartesian_to_eulerian, eulerian_to_cartesian,
                                  interpolate_quaternions,
                                  rotationBetweenVectors)

ROOT_FOLDER = join(config.RAWDATADIR, 'David_MMSys_18/dataset/')
OUTPUT_FOLDER = join(config.DATADIR, 'David_MMSys_18/sampled_dataset')
OUTPUT_TRUE_SALIENCY_FOLDER = join(config.DATADIR, 'David_MMSys_18/true_saliency')
SAMPLING_RATE = 0.2
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

VIDEOS = [
    '1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
    '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
    '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar',
    '19_Touvet'
]

# From "David_MMSys_18/dataset/Videos/Readme_Videos.md"
# Text files are provided with scanpaths from head movement with 100 samples per observer
NUM_SAMPLES_PER_USER = 100


def get_orientations_for_trace(filename) -> np.ndarray:
  dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
  data = dataframe[[' longitude', ' latitude']]
  return data.values


def get_time_stamps_for_trace(filename) -> np.ndarray:
  dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
  data = dataframe[' start timestamp']
  return data.values


# returns the frame rate of a video using openCV
# ToDo Copied (changed videoname to videoname+'_saliency' and video_path folder) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname) -> float:
  video_mp4 = videoname + '_saliency.mp4'
  video_path = join(ROOT_FOLDER, 'content/saliency', video_mp4)
  video = cv2.VideoCapture(video_path)
  # Find OpenCV version
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
  if int(major_ver) < 3:
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
  else:
    fps = video.get(cv2.CAP_PROP_FPS)
  video.release()
  return fps


# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset() -> dict:
  original_path = join(ROOT_FOLDER, 'Videos/H/Scanpaths')
  dataset = {}
  for root, directories, files in os.walk(original_path):
    for enum_trace, filename in enumerate(files):
      splitted_filename = filename.split('_')
      video = '_'.join(splitted_filename[:-1])
      file_path = join(root, filename)
      positions_all_users = get_orientations_for_trace(file_path)
      time_stamps_all_users = get_time_stamps_for_trace(file_path)
      num_users = int(positions_all_users.shape[0] / NUM_SAMPLES_PER_USER)
      for user_id in range(num_users):
        user = str(user_id)
        if user not in dataset.keys():
          dataset[user] = {}
        positions = positions_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) *
                                        (NUM_SAMPLES_PER_USER)]
        time_stamps = time_stamps_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) *
                                            (NUM_SAMPLES_PER_USER)]
        samples = []
        for pos, t_stamp in zip(positions, time_stamps):
          samples.append({'sec': t_stamp / 1000.0, 'yaw': pos[0], 'pitch': pos[1]})
        dataset[user][video] = samples
  return dataset


# From "dataset/Videos/Readme_Videos.md"
# Latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied according to the
# resolution of the desired equi-rectangular image output dimension).
# Participants started exploring omnidirectional contents either from an implicit longitudinal center
# (0-degrees and center of the equirectangular projection) or from the opposite longitude (180-degrees).
def transform_the_degrees_in_range(yaw, pitch) -> tuple:
  yaw = yaw * 2 * np.pi
  pitch = pitch * np.pi
  return yaw, pitch


# Performs the opposite transformation than transform_the_degrees_in_range
# Transform the yaw values from range [0, 2pi] to range [0, 1]
# Transform the pitch values from range [0, pi] to range [0, 1]
def transform_the_radians_to_original(yaw, pitch)-> tuple:
  yaw = yaw / (2 * np.pi)
  pitch = pitch / np.pi
  return yaw, pitch


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate) -> dict:
  dataset = {}
  for enum_user, user in enumerate(original_dataset.keys()):
    dataset[user] = {}
    for enum_video, video in enumerate(original_dataset[user].keys()):
      sample_orig = np.array([1, 0, 0])
      data_per_video = []
      for sample in original_dataset[user][video]:
        sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
        sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
        quat_rot = rotationBetweenVectors(sample_orig, sample_new)
        # append the quaternion to the list
        data_per_video.append([sample['sec'], quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3]])
        # update the values of time and sample
      # interpolate the quaternions to have a rate of 0.2 secs
      data_per_video = np.array(data_per_video)
      dataset[user][video] = interpolate_quaternions(data_per_video[:, 0],
                                                     data_per_video[:, 1:],
                                                     rate=rate)
  return dataset


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace) -> np.ndarray:
  angles_per_video = []
  orig_vec = np.array([1, 0, 0])
  for sample in quaternions_trace:
    quat_rot = Quaternion(sample[1:])
    sample_new = quat_rot.rotate(orig_vec)
    restored_yaw, restored_pitch = cartesian_to_eulerian(sample_new[0], sample_new[1],
                                                         sample_new[2])
    restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
    angles_per_video.append(np.array([restored_yaw, restored_pitch]))
  return np.array(angles_per_video)


def recover_original_angles_from_xyz_trace(xyz_trace) -> np.ndarray:
  angles_per_video = []
  for sample in xyz_trace:
    restored_yaw, restored_pitch = cartesian_to_eulerian(sample[1], sample[2], sample[3])
    restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
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
  return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)


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
      dataset[user][video] = recover_xyz_from_quaternions_trace(sampled_dataset[user][video])
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


# ToDo, transform in a class this is the main function of this file
def create_and_store_sampled_dataset() -> None:
  original_dataset = get_original_dataset()
  sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
  xyz_dataset = get_xyz_dataset(sampled_dataset)
  store_dataset(xyz_dataset, OUTPUT_FOLDER)


def create_and_store_true_saliency(sampled_dataset) -> None:
  if not exists(OUTPUT_TRUE_SALIENCY_FOLDER):
    makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)
  # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
  # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
  def from_position_to_tile_probability_cartesian(pos):
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
                                       np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False))
    yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
    pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
    yaw_grid = yaw_grid * 2 * np.pi
    pitch_grid = pitch_grid * np.pi
    x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
    great_circle_distance = np.arccos(
        np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
    gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
    return gaussian_orth

  for enum_video, video in enumerate(VIDEOS):
    real_saliency_for_video = []
    for x_i in range(NUM_SAMPLES_PER_USER):
      tileprobs_for_video_cartesian = []
      for user in sampled_dataset.keys():
        tileprobs_cartesian = from_position_to_tile_probability_cartesian(
            sampled_dataset[user][video][x_i, 1:])
        tileprobs_for_video_cartesian.append(tileprobs_cartesian)
      tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
      real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian,
                                       axis=0) / tileprobs_for_video_cartesian.shape[0]
      real_saliency_for_video.append(real_saliency_cartesian)
    real_saliency_for_video = np.array(real_saliency_for_video)

    true_sal_out_file = join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
    np.save(true_sal_out_file, real_saliency_for_video)


def load_sampled_dataset() -> dict:
  list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
  dataset = {}
  for video in list_of_videos:
    for user in [o for o in os.listdir(join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
      if user not in dataset.keys():
        dataset[user] = {}
      path = join(OUTPUT_FOLDER, video, user)
      data = pd.read_csv(path, header=None)
      dataset[user][video] = data.values
  return dataset
