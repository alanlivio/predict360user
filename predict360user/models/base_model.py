from abc import abstractclassmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.models import Model


class BaseModel(Model):

  @abstractclassmethod
  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    pass

  @abstractclassmethod
  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    pass


def metric_orth_dist_cartesian(position_a, position_b):
  # Transform into directional vector in Cartesian Coordinate System
  norm_a = tf.sqrt(tf.square(position_a[:, :, 0:1]) + tf.square(position_a[:, :, 1:2]) + tf.square(position_a[:, :, 2:3]))
  norm_b = tf.sqrt(tf.square(position_b[:, :, 0:1]) + tf.square(position_b[:, :, 1:2]) + tf.square(position_b[:, :, 2:3]))
  x_true = position_a[:, :, 0:1]/norm_a
  y_true = position_a[:, :, 1:2]/norm_a
  z_true = position_a[:, :, 2:3]/norm_a
  x_pred = position_b[:, :, 0:1]/norm_b
  y_pred = position_b[:, :, 1:2]/norm_b
  z_pred = position_b[:, :, 2:3]/norm_b
  # Finally compute orthodromic distance
  # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
  # To keep the values in bound between -1 and 1
  great_circle_distance = tf.acos(tf.maximum(tf.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
  return great_circle_distance


def metric_orth_dist_eulerian(true_position, pred_position) -> float:
  yaw_true = (true_position[:, :, 0:1] - 0.5) * 2 * np.pi
  pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
  # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
  yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2 * np.pi
  pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
  # Finally compute orthodromic distance
  delta_long = tf.abs(tf.atan2(tf.sin(yaw_true - yaw_pred), tf.cos(yaw_true - yaw_pred)))
  numerator = tf.sqrt(
      tf.pow(tf.cos(pitch_pred) * tf.sin(delta_long), 2.0) + tf.pow(
          tf.cos(pitch_true) * tf.sin(pitch_pred) -
          tf.sin(pitch_true) * tf.cos(pitch_pred) * tf.cos(delta_long), 2.0))
  denominator = tf.sin(pitch_true) * tf.sin(pitch_pred) + tf.cos(pitch_true) * tf.cos(
      pitch_pred) * tf.cos(delta_long)
  great_circle_distance = tf.abs(tf.atan2(numerator, denominator))
  return great_circle_distance


# This way we ensure that the network learns to predict the delta angle
def delta_angle_from_ori_mag_dir(values):
  orientation = values[0]
  magnitudes = values[1] / 2.0
  directions = values[2]
  # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
  motion = magnitudes * directions

  yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
  pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

  cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
  cond_correct = tf.cast(
      tf.logical_and(tf.less_equal(pitch_pred_wo_corr, 1.0),
                      tf.greater_equal(pitch_pred_wo_corr, 0.0)), tf.float32)
  cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

  pitch_pred = cond_above * (
      1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (
          -pitch_pred_wo_corr)
  yaw_pred = tf.math.mod(
      cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below *
      (yaw_pred_wo_corr - 0.5), 1.0)
  return tf.concat([yaw_pred, pitch_pred], -1)