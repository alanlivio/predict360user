from head_motion_prediction.Utils import *
from spherical_geometry import polygon
import numpy as np
from numpy.typing import NDArray

class FOV:

    X1Y0Z0 = np.array([1, 0, 0])
    HOR_DIST = degrees_to_radian(110)
    HOR_MARGIN = degrees_to_radian(110/2)
    VER_MARGIN = degrees_to_radian(90/2)
    _x1y0z0_poly = None
    _x1y0z0_area = None
    _saved_points = None
    _saved_polys = None

    @classmethod
    @property
    def x1y0z0_poly(cls) -> NDArray:
        if cls._x1y0z0_poly is None:
            x1y0z0_fov_points_euler = np.array([
                eulerian_in_range(-FOV.HOR_MARGIN, FOV.VER_MARGIN),
                eulerian_in_range(FOV.HOR_MARGIN, FOV.VER_MARGIN),
                eulerian_in_range(FOV.HOR_MARGIN, -FOV.VER_MARGIN),
                eulerian_in_range(-FOV.HOR_MARGIN, -FOV.VER_MARGIN)
            ])
            cls._x1y0z0_poly = np.array([
                eulerian_to_cartesian(*x1y0z0_fov_points_euler[0]),
                eulerian_to_cartesian(*x1y0z0_fov_points_euler[1]),
                eulerian_to_cartesian(*x1y0z0_fov_points_euler[2]),
                eulerian_to_cartesian(*x1y0z0_fov_points_euler[3])
            ])
        return cls._x1y0z0_poly
        
    @classmethod
    def x1y0z0_area(cls):
        if cls._x1y0z0_area is None:
            cls._x1y0z0_area = polygon.SphericalPolygon(cls.x1y0z0_poly).area()
        
    @classmethod
    def points(cls, trace) -> NDArray:
        if cls._saved_points is None:
            cls._saved_points = {}
        if (trace[0], trace[1], trace[2]) not in cls._saved_points:
            rotation = rotationBetweenVectors(cls.X1Y0Z0, np.array(trace))
            points = np.array([
                rotation.rotate(cls.x1y0z0_poly[0]),
                rotation.rotate(cls.x1y0z0_poly[1]),
                rotation.rotate(cls.x1y0z0_poly[2]),
                rotation.rotate(cls.x1y0z0_poly[3]),
            ])
            cls._saved_points[(trace[0], trace[1], trace[2])] =  points
        return cls._saved_points[(trace[0], trace[1], trace[2])]

    @classmethod
    def poly(cls, trace) -> polygon.SphericalPolygon:
        if cls._saved_polys is None:
            cls._saved_polys = {}
        if (trace[0], trace[1], trace[2]) not in cls._saved_polys:
            points_trace = cls.points(trace)
            cls._saved_polys[(trace[0], trace[1], trace[2])] = polygon.SphericalPolygon(points_trace)
        return cls._saved_polys[(trace[0], trace[1], trace[2])]
