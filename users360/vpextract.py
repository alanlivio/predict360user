from head_motion_prediction.Utils import *
from abc import abstractmethod
from enum import Enum, auto
from scipy.spatial import SphericalVoronoi
from typing import Tuple
from spherical_geometry import polygon
from abc import ABC
import math
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objs as go

VP_MARGIN_THETA = degrees_to_radian(110/2)
VP_MARGIN_THI = degrees_to_radian(90/2)
X1Y0Z0_POLYG_FOV = np.array([
    eulerian_in_range(-VP_MARGIN_THETA, VP_MARGIN_THI),
    eulerian_in_range(VP_MARGIN_THETA, VP_MARGIN_THI),
    eulerian_in_range(VP_MARGIN_THETA, -VP_MARGIN_THI),
    eulerian_in_range(-VP_MARGIN_THETA, -VP_MARGIN_THI)
])
X1Y0Z0_POLYG_FOV = np.array([
    eulerian_to_cartesian(*X1Y0Z0_POLYG_FOV[0]),
    eulerian_to_cartesian(*X1Y0Z0_POLYG_FOV[1]),
    eulerian_to_cartesian(*X1Y0Z0_POLYG_FOV[2]),
    eulerian_to_cartesian(*X1Y0Z0_POLYG_FOV[3])
])
X1Y0Z0 = np.array([1, 0, 0])
X1Y0Z0_POLYG_FOV_AREA = polygon.SphericalPolygon(X1Y0Z0_POLYG_FOV, inside=X1Y0Z0).area()

_saved_fov_points = {}
def fov_points(trace) -> NDArray:
    if (trace[0], trace[1], trace[2]) not in _saved_fov_points:
        rotation = rotationBetweenVectors(X1Y0Z0, np.array(trace))
        points = np.array([
            rotation.rotate(X1Y0Z0_POLYG_FOV[0]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[1]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[2]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[3]),
        ])
        _saved_fov_points[(trace[0], trace[1], trace[2])] =  points
    return _saved_fov_points[(trace[0], trace[1], trace[2])]

_saved_fov_polys = {}
def fov_poly(trace) -> polygon.SphericalPolygon:
    if (trace[0], trace[1], trace[2]) not in _saved_fov_polys:
        fov_points_trace = fov_points(trace)
        _saved_fov_polys[(trace[0], trace[1], trace[2])] = polygon.SphericalPolygon(np.unique(fov_points_trace,axis=0), inside=trace)
    return _saved_fov_polys[(trace[0], trace[1], trace[2])]

class VPExtract(ABC):
    class Cover(Enum):
        ANY = auto()
        CENTER = auto()
        ONLY20PERC = auto()
        ONLY33PERC = auto()

    @abstractmethod
    def request(self, trace, return_metrics=False) -> tuple[NDArray, float, list]:
        pass

    @property
    def title(self):
        pass

    def title_with_sum_heatmaps(self, heatmaps):
        reqs_sum = np.sum(np.sum(heatmaps, axis=0))
        return f"{self.title} (reqs={reqs_sum})"

    cover: Cover
    shape: tuple[int, int]


_saved_tile_polys = {}
_saved_tile_centers = {}

def tile_points(t_ver, t_hor, row, col) -> NDArray:
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

def _saved_tile_init(t_ver, t_hor):
    d_hor = degrees_to_radian(360/t_hor)
    d_ver = degrees_to_radian(180/t_ver)
    polys, centers = {}, {}
    for row in range(t_ver):
        polys[row], centers[row] = {}, {}
        for col in range(t_hor):
            polys[row][col] = polygon.SphericalPolygon(tile_points(t_ver, t_hor, row, col))
            theta_c = d_hor * (col + 0.5)
            phi_c = d_ver * (row + 0.5)
            # at eulerian_to_cartesian: theta is hor and phi is ver 
            centers[row][col] = eulerian_to_cartesian(theta_c, phi_c)
    _saved_tile_polys[(t_ver, t_hor)] = polys
    _saved_tile_centers[(t_ver, t_hor)] = centers

def tile_poly(t_ver, t_hor, row, col):
    if (t_ver, t_hor) not in _saved_tile_polys:
        _saved_tile_init(t_ver, t_hor)
    return _saved_tile_polys[(t_ver, t_hor)][row][col]

def tile_center(t_ver, t_hor, row, col):
    if (t_ver, t_hor) not in _saved_tile_polys:
        _saved_tile_init(t_ver, t_hor)
    return _saved_tile_centers[(t_ver, t_hor)][row][col]

_vp_55d_rad = degrees_to_radian(110/2)
_vp_110d_rad = degrees_to_radian(110)

class VPExtractTilesRect(VPExtract):
    
    def __init__(self, t_ver, t_hor, cover: VPExtract.Cover):
        self.t_ver, self.t_hor = t_ver, t_hor
        self.shape = (self.t_ver, self.t_hor)
        self.cover = cover

    @property
    def title(self):
        prefix = f'vpextract_rect{self.t_ver}x{self.t_hor}'
        match self.cover:
            case VPExtract.Cover.ANY:
                return f'{prefix}_any'
            case VPExtract.Cover.CENTER:
                return f'{prefix}_center'
            case VPExtract.Cover.ONLY20PERC:
                return f'{prefix}_20perc'
            case VPExtract.Cover.ONLY33PERC:
                return f'{prefix}_33perc'

    def request(self, trace: NDArray, return_metrics=False):
        match self.cover:
            case VPExtract.Cover.CENTER:
                return self._request_110radius_center(trace, return_metrics)
            case VPExtract.Cover.ANY:
                return self._request_min_cover(trace, 0.0, return_metrics)
            case VPExtract.Cover.ONLY20PERC:
                return self._request_min_cover(trace, 0.2, return_metrics)
            case VPExtract.Cover.ONLY33PERC:
                return self._request_min_cover(trace, 0.33, return_metrics)

    def _request_110radius_center(self, trace, return_metrics):
        heatmap = np.zeros((self.t_ver, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = fov_poly(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, tile_center(self.t_ver, self.t_hor, row, col))
                if dist <= _vp_55d_rad:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        try:
                            tile_poly_rc = tile_poly(self.t_ver, self.t_hor, row, col)
                            view_ratio = tile_poly_rc.overlap(fov_poly_trace)
                        except:
                            print(f"request error for row,col,trace={row},{col},{repr(trace)}")
                            continue
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly_trace.overlap(tile_poly_rc)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace: NDArray, required_cover: float, return_metrics):
        heatmap = np.zeros((self.t_ver, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = fov_poly(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, tile_center(self.t_ver, self.t_hor, row, col))
                if dist >= _vp_110d_rad:
                    continue
                try:
                    tile_poly_rc = tile_poly(self.t_ver, self.t_hor, row, col)
                    view_ratio = tile_poly_rc.overlap(fov_poly_trace)
                except:
                    print(f"request error for row,col,trace={row},{col},{repr(trace)}")
                    continue
                if view_ratio > required_cover:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly_trace.overlap(tile_poly_rc)
        return heatmap, vp_quality, np.sum(areas_out)


def create_trinity_voro(npatchs) -> SphericalVoronoi:
    points = np.empty((0, 3))
    for index in range(0, npatchs):
        zi = (1 - 1.0/npatchs) * (1 - 2.0*index / (npatchs - 1))
        di = math.sqrt(1 - math.pow(zi, 2))
        alphai = index * math.pi * (3 - math.sqrt(5))
        xi = di * math.cos(alphai)
        yi = di * math.sin(alphai)
        new_point = np.array([[xi, yi, zi]])
        points = np.append(points, new_point, axis=0)
    sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    return sv
VORONOI_14P = create_trinity_voro(14)
VORONOI_24P = create_trinity_voro(24)

_saved_tilevoro_polys = {}

class VPExtractTilesVoro(VPExtract):
    def __init__(self, sphere_voro: SphericalVoronoi, cover: VPExtract.Cover):
        super().__init__()
        self.sphere_voro = sphere_voro
        self.cover = cover
        self.shape = (2, -1)
        if sphere_voro.points.size not in _saved_tilevoro_polys:
            self.voro_tile_polys = {index: polygon.SphericalPolygon(self.sphere_voro.vertices[self.sphere_voro.regions[index]]) for index, _ in enumerate(self.sphere_voro.regions)}
            _saved_tilevoro_polys[sphere_voro.points.size] = self.voro_tile_polys
        else:
            self.voro_tile_polys = _saved_tilevoro_polys[sphere_voro.points.size]
        
    @property
    def title(self):
        prefix = f'vpextract_voro{len(self.sphere_voro.points)}'
        match self.cover:
            case VPExtract.Cover.ANY:
                return f'{prefix}_any'
            case VPExtract.Cover.CENTER:
                return f'{prefix}_center'
            case VPExtract.Cover.ONLY20PERC:
                return f'{prefix}_20perc'
            case VPExtract.Cover.ONLY33PERC:
                return f'{prefix}_33perc'

    def request(self, trace, return_metrics=False):
        match self.cover:
            case VPExtract.Cover.CENTER:
                return self._request_110radius_center(trace, return_metrics)
            case VPExtract.Cover.ANY:
                return self._request_min_cover(trace, 0, return_metrics)
            case VPExtract.Cover.ONLY20PERC:
                return self._request_min_cover(trace, 0.2, return_metrics)
            case VPExtract.Cover.ONLY33PERC:
                return self._request_min_cover(trace, 0.33, return_metrics)

    def _request_110radius_center(self, trace, return_metrics):
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = fov_poly(trace)
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, _ in enumerate(self.sphere_voro.regions):
            dist = compute_orthodromic_distance(trace, self.sphere_voro.points[index])
            if dist <= _vp_55d_rad:
                heatmap[index] += 1
                if(return_metrics):
                    voro_tile_poly = self.voro_tile_polys[index]
                    view_ratio = voro_tile_poly.overlap(fov_poly_trace)
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly_trace.overlap(voro_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace, required_cover: float, return_metrics):
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = fov_poly(trace)
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, _ in enumerate(self.sphere_voro.regions):
            voro_tile_poly = self.voro_tile_polys[index]
            view_ratio = voro_tile_poly.overlap(fov_poly_trace)
            if view_ratio > required_cover:
                heatmap[index] += 1
                if(return_metrics):
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly_trace.overlap(voro_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)


VPEXTRACT_RECT_4_6_CENTER = VPExtractTilesRect(4, 6,VPExtract.Cover.CENTER)
VPEXTRACT_RECT_4_6_ANY = VPExtractTilesRect(4, 6,VPExtract.Cover.ANY)
VPEXTRACT_RECT_4_6_20PERC = VPExtractTilesRect(4, 6,VPExtract.Cover.ONLY20PERC)
VPEXTRACT_VORO_14_CENTER = VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.CENTER)
VPEXTRACT_VORO_14_ANY = VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ANY)
VPEXTRACT_VORO_14_20PERC = VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ONLY20PERC)
# VPEXTRACT_VORO_24_CENTER = VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.CEMTER)
# VPEXTRACT_VORO_24_ANY = VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ANY)
# VPEXTRACT_VORO_24_20PERC = VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ONLY20PERC)

VPEXTRACTS_VORO = [
    VPEXTRACT_VORO_14_CENTER,
    VPEXTRACT_VORO_14_ANY,
    VPEXTRACT_VORO_14_20PERC,
]
VPEXTRACTS_RECT = [
    VPEXTRACT_RECT_4_6_CENTER,
    VPEXTRACT_RECT_4_6_ANY,
    VPEXTRACT_RECT_4_6_20PERC,
]
VPEXTRACT_METHODS = [*VPEXTRACTS_VORO, *VPEXTRACTS_RECT]