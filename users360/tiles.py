from head_motion_prediction.Utils import *
from abc import abstractmethod
from enum import Enum, auto
from spherical_geometry import polygon
from abc import ABC
import numpy as np
from numpy.typing import NDArray
from .fov import *

class TileCover(Enum):
    ANY = auto()
    CENTER = auto()
    ONLY20PERC = auto()
    ONLY33PERC = auto()
    
class TilesIF(ABC):
    # https://realpython.com/python-interface/
    
    cover: TileCover
    shape: tuple[int, int]
    
    @abstractmethod
    def request(self, trace, return_metrics=False) -> tuple[NDArray, float, list]:
        pass

    @property
    def title(self):
        pass

    def title_with_sum_heatmaps(self, heatmaps):
        reqs_sum = np.sum(np.sum(heatmaps, axis=0))
        return f"{self.title} (reqs={reqs_sum})"


class Tiles(TilesIF):
    
    def __init__(self, t_ver, t_hor, cover: TileCover):
        self.t_ver, self.t_hor = t_ver, t_hor
        self.shape = (self.t_ver, self.t_hor)
        self.cover = cover

    _default = None
    @classmethod
    def default(cls):
        if cls._default is None:
            cls._default = Tiles(4, 6,TileCover.CENTER)
        return cls._default
    
    _variations = None
    @classmethod
    def variations(cls):
        if cls._variations is None:
            cls._variations = [
            Tiles(4, 6,TileCover.CENTER),
            Tiles(4, 6,TileCover.ANY),
            Tiles(4, 6,TileCover.ONLY20PERC),
        ]
        return cls._variations

    @property
    def title(self):
        prefix = f'tiles_{self.t_ver}x{self.t_hor}'
        match self.cover:
            case TileCover.ANY:
                return f'{prefix}_any'
            case TileCover.CENTER:
                return f'{prefix}_center'
            case TileCover.ONLY20PERC:
                return f'{prefix}_20perc'
            case TileCover.ONLY33PERC:
                return f'{prefix}_33perc'

    def request(self, trace: NDArray, return_metrics=False):
        match self.cover:
            case TileCover.CENTER:
                return self._request_110radius_center(trace, return_metrics)
            case TileCover.ANY:
                return self._request_min_cover(trace, 0.0, return_metrics)
            case TileCover.ONLY20PERC:
                return self._request_min_cover(trace, 0.2, return_metrics)
            case TileCover.ONLY33PERC:
                return self._request_min_cover(trace, 0.33, return_metrics)


    @classmethod
    def tile_points(cls,t_ver, t_hor, row, col) -> NDArray:
        d_ver = degrees_to_radian(180/t_ver)
        d_hor = degrees_to_radian(360/t_hor)
        assert (row < t_ver and col < t_hor)
        points = np.array([
            eulerian_to_cartesian(d_hor * col,      d_ver * (row+1)),
            eulerian_to_cartesian(d_hor * (col+1),  d_ver * (row+1)),
            eulerian_to_cartesian(d_hor * (col+1),  d_ver * row),
            eulerian_to_cartesian(d_hor * col,      d_ver * row),
        ])
        return points


    _saved_polys = None
    _saved_centers = None

    @classmethod
    def _saved_tiles_init(cls, t_ver, t_hor):
        cls._saved_polys = {}
        cls._saved_centers = {}
        d_hor = degrees_to_radian(360/t_hor)
        d_ver = degrees_to_radian(180/t_ver)
        polys, centers = {}, {}
        for row in range(t_ver):
            polys[row], centers[row] = {}, {}
            for col in range(t_hor):
                points = cls.tile_points(t_ver, t_hor, row, col)
                # sometimes tile points are same on poles
                # they need be unique otherwise it broke the SphericalPolygon.area
                _, idx = np.unique(points,axis=0, return_index=True)
                polys[row][col] = polygon.SphericalPolygon(points[np.sort(idx)])
                theta_c = d_hor * (col + 0.5)
                phi_c = d_ver * (row + 0.5)
                # at eulerian_to_cartesian: theta is hor and phi is ver 
                centers[row][col] = eulerian_to_cartesian(theta_c, phi_c)
        cls._saved_polys[(t_ver, t_hor)] = polys
        cls._saved_centers[(t_ver, t_hor)] = centers

    @classmethod
    def tile_poly(cls, t_ver, t_hor, row, col):
        if cls._saved_polys is None or (t_ver, t_hor) not in cls._saved_polys:
            cls._saved_tiles_init(t_ver, t_hor)
        return cls._saved_polys[(t_ver, t_hor)][row][col]

    @classmethod
    def tile_center(cls, t_ver, t_hor, row, col):
        if cls._saved_polys is None or (t_ver, t_hor) not in cls._saved_polys:
            cls._saved_tiles_init(t_ver, t_hor)
        return cls._saved_centers[(t_ver, t_hor)][row][col]

    def _request_110radius_center(self, trace, return_metrics):
        heatmap = np.zeros((self.t_ver, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = FOV.poly(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, Tiles.tile_center(self.t_ver, self.t_hor, row, col))
                if dist <= FOV.HOR_MARGIN:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        try:
                            poly_rc = Tiles.tile_poly(self.t_ver, self.t_hor, row, col)
                            view_ratio = poly_rc.overlap(fov_poly_trace)
                        except:
                            print(f"request error for row,col,trace={row},{col},{repr(trace)}")
                            continue
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly_trace.overlap(poly_rc)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace: NDArray, required_cover: float, return_metrics):
        heatmap = np.zeros((self.t_ver, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = FOV.poly(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, Tiles.tile_center(self.t_ver, self.t_hor, row, col))
                if dist >= FOV.HOR_DIST:
                    continue
                try:
                    poly_rc = Tiles.tile_poly(self.t_ver, self.t_hor, row, col)
                    view_ratio = poly_rc.overlap(fov_poly_trace)
                except:
                    print(f"request error for row,col,trace={row},{col},{repr(trace)}")
                    continue
                if view_ratio > required_cover:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly_trace.overlap(poly_rc)
        return heatmap, vp_quality, np.sum(areas_out)