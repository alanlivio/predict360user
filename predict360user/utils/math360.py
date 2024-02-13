from ast import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from numpy import cross, dot
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.preprocessing import normalize


def degrees_to_radian(degree):
    return degree * np.pi / 180.0


def radian_to_degrees(radian):
    return radian * 180.0 / np.pi


def orthogonal(v):
    x = abs(v[0])
    y = abs(v[1])
    z = abs(v[2])
    other = (1, 0, 0) if (x < y and x < z) else (0, 1, 0) if (y < z) else (0, 0, 1)
    return cross(v, other)


def normalized(v):
    return normalize(v[:, np.newaxis], axis=0).ravel()


# Compute the orthodromic distance between two points in 3d coordinates
def orth_dist_cartesian(position_a, position_b):
    norm_a = np.sqrt(
        np.square(position_a[0]) + np.square(position_a[1]) + np.square(position_a[2])
    )
    norm_b = np.sqrt(
        np.square(position_b[0]) + np.square(position_b[1]) + np.square(position_b[2])
    )
    x_true = position_a[0] / norm_a
    y_true = position_a[1] / norm_a
    z_true = position_a[2] / norm_a
    x_pred = position_b[0] / norm_b
    y_pred = position_b[1] / norm_b
    z_pred = position_b[2] / norm_b
    great_circle_distance = np.arccos(
        np.maximum(
            np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0
        )
    )
    return great_circle_distance


# The (input) corresponds to (x, y, z) of a unit sphere centered at the origin (0, 0, 0)
# Returns the values (theta, phi) with:
# theta in the range 0, to 2*pi, theta can be negative, e.g. cartesian_to_eulerian(0, -1, 0) = (-pi/2, pi/2) (is equal to (3*pi/2, pi/2))
# phi in the range 0 to pi (0 being the north pole, pi being the south pole)
def cartesian_to_eulerian(x, y, z) -> tuple[float, float]:
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    # remainder is used to transform it in the positive range (0, 2*pi)
    theta = np.remainder(theta, 2 * np.pi)
    return theta, phi


# The (input) values of theta and phi are assumed to be as follows:
# theta = Any              phi =   0    : north pole (0, 0, 1)
# theta = Any              phi =  pi    : south pole (0, 0, -1)
# theta = 0, 2*pi          phi = pi/2   : equator facing (1, 0, 0)
# theta = pi/2             phi = pi/2   : equator facing (0, 1, 0)
# theta = pi               phi = pi/2   : equator facing (-1, 0, 0)
# theta = -pi/2, 3*pi/2    phi = pi/2   : equator facing (0, -1, 0)
# In other words
# The longitude ranges from 0, to 2*pi
# The latitude ranges from 0 to pi, origin of equirectangular in the top-left corner
# Returns the values (x, y, z) of a unit sphere with center in (0, 0, 0)
def eulerian_to_cartesian(theta, phi) -> np.ndarray:
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])


# Transforms the eulerian angles from range (0, 2*pi) and (0, pi) to (-pi, pi) and (-pi/2, pi/2)
def eulerian_in_range(theta, phi) -> tuple[float, float]:
    theta = theta - np.pi
    phi = phi - (np.pi / 2.0)
    return theta, phi


def metric_orth_dist_cartesian(positions_a, positions_b) -> float:
    # Transform into directional vector in Cartesian Coordinate System
    norm_a = tf.sqrt(
        tf.square(positions_a[:, :, 0:1])
        + tf.square(positions_a[:, :, 1:2])
        + tf.square(positions_a[:, :, 2:3])
    )
    norm_b = tf.sqrt(
        tf.square(positions_b[:, :, 0:1])
        + tf.square(positions_b[:, :, 1:2])
        + tf.square(positions_b[:, :, 2:3])
    )
    x_true = positions_a[:, :, 0:1] / norm_a
    y_true = positions_a[:, :, 1:2] / norm_a
    z_true = positions_a[:, :, 2:3] / norm_a
    x_pred = positions_b[:, :, 0:1] / norm_b
    y_pred = positions_b[:, :, 1:2] / norm_b
    z_pred = positions_b[:, :, 2:3] / norm_b
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = tf.acos(
        tf.maximum(
            tf.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0
        )
    )
    return great_circle_distance


def metric_orth_dist_eulerian(positions_a, positions_b) -> float:
    yaw_true = (positions_a[:, :, 0:1] - 0.5) * 2 * np.pi
    pitch_true = (positions_a[:, :, 1:2] - 0.5) * np.pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (positions_b[:, :, 0:1] - 0.5) * 2 * np.pi
    pitch_pred = (positions_b[:, :, 1:2] - 0.5) * np.pi
    # Finally compute orthodromic distance
    delta_long = tf.abs(
        tf.atan2(tf.sin(yaw_true - yaw_pred), tf.cos(yaw_true - yaw_pred))
    )
    numerator = tf.sqrt(
        tf.pow(tf.cos(pitch_pred) * tf.sin(delta_long), 2.0)
        + tf.pow(
            tf.cos(pitch_true) * tf.sin(pitch_pred)
            - tf.sin(pitch_true) * tf.cos(pitch_pred) * tf.cos(delta_long),
            2.0,
        )
    )
    denominator = tf.sin(pitch_true) * tf.sin(pitch_pred) + tf.cos(pitch_true) * tf.cos(
        pitch_pred
    ) * tf.cos(delta_long)
    great_circle_distance = tf.abs(tf.atan2(numerator, denominator))
    return great_circle_distance


def rotationBetweenVectors(u, v) -> Quaternion:
    u = normalized(u)
    v = normalized(v)
    if np.allclose(u, v):
        return Quaternion(angle=0.0, axis=u)
    if np.allclose(u, -v):
        return Quaternion(angle=np.pi, axis=normalized(orthogonal(u)))
    quat = Quaternion(angle=np.arccos(dot(u, v)), axis=normalized(cross(u, v)))
    return quat


X1Y0Z0 = np.array([1, 0, 0])
HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
RES_WIDTH = 3840
RES_HEIGHT = 2160

_fov_x1y0z0_fov_points_euler = np.array(
    [
        eulerian_in_range(-HOR_MARGIN, VER_MARGIN),
        eulerian_in_range(HOR_MARGIN, VER_MARGIN),
        eulerian_in_range(HOR_MARGIN, -VER_MARGIN),
        eulerian_in_range(-HOR_MARGIN, -VER_MARGIN),
    ]
)
_fov_x1y0z0_points = np.array(
    [
        eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[0]),
        eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[1]),
        eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[2]),
        eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[3]),
    ]
)


def fov_points(x, y, z) -> np.ndarray:
    rotation = rotationBetweenVectors(X1Y0Z0, np.array([x, y, z]))
    points = np.array(
        [
            rotation.rotate(_fov_x1y0z0_points[0]),
            rotation.rotate(_fov_x1y0z0_points[1]),
            rotation.rotate(_fov_x1y0z0_points[2]),
            rotation.rotate(_fov_x1y0z0_points[3]),
        ]
    )
    return points


def calc_fixmps_ids(traces: np.ndarray) -> np.ndarray:
    # calc fixation_ids
    scale = 0.025
    n_height = int(scale * RES_HEIGHT)
    n_width = int(scale * RES_WIDTH)
    im_theta = np.linspace(0, 2 * np.pi - 2 * np.pi / n_width, n_width, endpoint=True)
    im_phi = np.linspace(
        0 + np.pi / (2 * n_height),
        np.pi - np.pi / (2 * n_height),
        n_height,
        endpoint=True,
    )

    def calc_one_fixmap_id(trace) -> np.int64:
        fixmp = np.zeros((n_height, n_width))
        target_theta, target_thi = cartesian_to_eulerian(*trace)
        mindiff_theta = np.min(abs(im_theta - target_theta))
        im_col = np.where(np.abs(im_theta - target_theta) == mindiff_theta)[0][0]
        mindiff_phi = min(abs(im_phi - target_thi))
        im_row = np.where(np.abs(im_phi - target_thi) == mindiff_phi)[0][0]
        fixmp[im_row, im_col] = 1
        fixmp_id = np.nonzero(fixmp.reshape(-1))[0][0]
        assert isinstance(fixmp_id, np.int64)
        return fixmp_id

    fixmps_ids = np.apply_along_axis(calc_one_fixmap_id, 1, traces)
    assert fixmps_ids.shape == (len(traces),)
    return fixmps_ids


def calc_actual_entropy_from_ids(x_ids_t: np.ndarray, return_sub_len_t=False) -> Union[float, Tuple]:
    assert isinstance(x_ids_t, np.ndarray)
    n = len(x_ids_t)
    sub_len_l = np.zeros(n)
    sub_len_l[0] = 1
    for i in range(1, n):
        # sub_1st as current i
        sub_1st = x_ids_t[i]
        # case sub_1st not seen, so set 1
        sub_len_l[i] = 1
        sub_1st_seen_idxs = np.nonzero(x_ids_t[0:i] == sub_1st)[0]
        if sub_1st_seen_idxs.size == 0:
            continue
        # case sub_1st seen, search longest valid k-lengh sub
        for idx in sub_1st_seen_idxs:
            k = 1
            while i + k < n and idx + k <= i:  # skip the last  # until previous i
                # given valid set current k if longer
                sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
                # try match with k-lengh from idx
                next_sub = x_ids_t[i : i + k]
                k_sub = x_ids_t[idx : idx + k]
                # if not match with k-lengh from idx
                if not np.array_equal(next_sub, k_sub):
                    break
                # if match increase k and set if longer
                k += 1
                sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
    actual_entropy = (1 / ((1 / n) * np.sum(sub_len_l))) * np.log2(n)
    actual_entropy = np.round(actual_entropy, 3)
    if return_sub_len_t:
        return actual_entropy, sub_len_l
    else:
        return actual_entropy


def calc_actual_entropy(traces: np.ndarray) -> float:
    fixmps_ids = calc_fixmps_ids(traces)
    return calc_actual_entropy_from_ids(fixmps_ids)


# time_orig_at_zero is a flag to determine if the time must start counting from zero, if so, the trace is forced to start at 0.0
def interpolate_quaternions(
    orig_times, quaternions, rate, time_orig_at_zero=True
) -> np.ndarray:
    # if the first time-stamps is greater than (half) the frame rate, put the time-stamp 0.0 and copy the first quaternion to the beginning
    if time_orig_at_zero and (orig_times[0] > rate / 2.0):
        orig_times = np.concatenate(([0.0], orig_times))
        # ToDo use the quaternion rotation to predict where the position was at t=0
        quaternions = np.concatenate(([quaternions[0]], quaternions))
    key_rots = R.from_quat(quaternions)
    slerp = Slerp(orig_times, key_rots)
    # we add rate/2 to the last time-stamp so we include it in the possible interpolation time-stamps
    times = np.arange(orig_times[0], orig_times[-1] + rate / 2.0, rate)
    # to bound it to the maximum original-time in the case of rounding errors
    times[-1] = min(orig_times[-1], times[-1])
    interp_rots = slerp(times)
    return np.concatenate((times[:, np.newaxis], interp_rots.as_quat()), axis=1)
