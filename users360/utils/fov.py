from functools import cache

import numpy as np
from spherical_geometry import polygon

from .. import config
from ..head_motion_prediction.Utils import (cartesian_to_eulerian,
                                            degrees_to_radian,
                                            eulerian_in_range,
                                            eulerian_to_cartesian,
                                            rotationBetweenVectors)

X1Y0Z0 = np.array([1, 0, 0])
HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
RES_WIDTH = 3840
RES_HIGHT = 2160

_fov_x1y0z0_fov_points_euler = np.array([
    eulerian_in_range(-HOR_MARGIN, VER_MARGIN),
    eulerian_in_range(HOR_MARGIN, VER_MARGIN),
    eulerian_in_range(HOR_MARGIN, -VER_MARGIN),
    eulerian_in_range(-HOR_MARGIN, -VER_MARGIN)
])
_fov_x1y0z0_points = np.array([
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[0]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[1]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[2]),
    eulerian_to_cartesian(*_fov_x1y0z0_fov_points_euler[3])
])


@cache
def fov_points(x, y, z) -> np.ndarray:
  rotation = rotationBetweenVectors(X1Y0Z0, np.array([x, y, z]))
  points = np.array([
      rotation.rotate(_fov_x1y0z0_points[0]),
      rotation.rotate(_fov_x1y0z0_points[1]),
      rotation.rotate(_fov_x1y0z0_points[2]),
      rotation.rotate(_fov_x1y0z0_points[3]),
  ])
  return points


@cache
def fov_poly(x, y, z) -> polygon.SphericalPolygon:
  points_trace = fov_points(x, y, z)
  return polygon.SphericalPolygon(points_trace)


def calc_fixmps_ids(traces: np.array) -> np.array:
  # calc fixation_ids
  scale = 0.025
  n_height = int(scale * RES_HIGHT)
  n_width = int(scale * RES_WIDTH)
  im_theta = np.linspace(0, 2 * np.pi - 2 * np.pi / n_width, n_width, endpoint=True)
  im_phi = np.linspace(0 + np.pi / (2 * n_height),
                       np.pi - np.pi / (2 * n_height),
                       n_height,
                       endpoint=True)

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
  assert fixmps_ids.shape == (len(traces), )
  return fixmps_ids


def calc_actual_entropy_from_ids(x_ids_t: np.ndarray, return_sub_len_t=False) -> float:
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
      while (i + k < n  # skip the last
             and idx + k <= i  # until previous i
             ):
        # given valid set current k if longer
        sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
        # try match with k-lengh from idx
        next_sub = x_ids_t[i:i + k]
        k_sub = x_ids_t[idx:idx + k]
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

def calc_actual_entropy(traces: np.array) -> float:
  fixmps_ids = calc_fixmps_ids(traces)
  return calc_actual_entropy_from_ids(fixmps_ids)