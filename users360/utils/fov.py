import numpy as np
from spherical_geometry import polygon

from ..head_motion_prediction.Utils import *

X1Y0Z0 = np.array([1, 0, 0])
HOR_DIST = degrees_to_radian(110)
HOR_MARGIN = degrees_to_radian(110 / 2)
VER_MARGIN = degrees_to_radian(90 / 2)
_fov_points = dict()
_fov_polys = dict()

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


def fov_points(trace) -> np.ndarray:
    if (trace[0], trace[1], trace[2]) not in _fov_points:
        rotation = rotationBetweenVectors(X1Y0Z0, np.array(trace))
        points = np.array([
            rotation.rotate(_fov_x1y0z0_points[0]),
            rotation.rotate(_fov_x1y0z0_points[1]),
            rotation.rotate(_fov_x1y0z0_points[2]),
            rotation.rotate(_fov_x1y0z0_points[3]),
        ])
        _fov_points[(trace[0], trace[1], trace[2])] = points
    return _fov_points[(trace[0], trace[1], trace[2])]


def fov_poly(trace) -> polygon.SphericalPolygon:
    if (trace[0], trace[1], trace[2]) not in _fov_polys:
        points_trace = fov_points(trace)
        _fov_polys[(trace[0], trace[1], trace[2])] = polygon.SphericalPolygon(points_trace)
    return _fov_polys[(trace[0], trace[1], trace[2])]
