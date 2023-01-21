from functools import cache

import numpy as np
from spherical_geometry import polygon

from ..head_motion_prediction.Utils import (degrees_to_radian,
                                            eulerian_in_range,
                                            eulerian_to_cartesian,
                                            rotationBetweenVectors)

X1Y0Z0 = np.array([1, 0, 0])
HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)

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
