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

def points_rect_tile_cartesian(row, col, t_ver, t_hor) -> NDArray:
    d_hor = degrees_to_radian(360/t_hor)
    d_vert = degrees_to_radian(180/t_ver)
    assert (row < t_ver and col < t_hor)
    polygon_rect_tile = np.array([
        eulerian_to_cartesian(d_hor * col, d_vert * (row+1)),
        eulerian_to_cartesian(d_hor * (col+1), d_vert * (row+1)),
        eulerian_to_cartesian(d_hor * (col+1), d_vert * row),
        eulerian_to_cartesian(d_hor * col, d_vert * row)])
    return polygon_rect_tile

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

saved_points_fov_cartesian = {}
def points_fov_cartesian(trace) -> NDArray:
    if (trace[0], trace[1], trace[2]) not in saved_points_fov_cartesian:
        rotation = rotationBetweenVectors(X1Y0Z0, np.array(trace))
        poly_fov = np.array([
            rotation.rotate(X1Y0Z0_POLYG_FOV[0]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[1]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[2]),
            rotation.rotate(X1Y0Z0_POLYG_FOV[3]),
        ])
        saved_points_fov_cartesian[(trace[0], trace[1], trace[2])] =  poly_fov
    return saved_points_fov_cartesian[(trace[0], trace[1], trace[2])]

saved_poly_fov = {}
def poly_fov(trace) -> polygon.SphericalPolygon:
    if (trace[0], trace[1], trace[2]) not in saved_poly_fov:
        fov_car = points_fov_cartesian(trace)
        saved_poly_fov[(trace[0], trace[1], trace[2])] = polygon.SphericalPolygon(np.unique(fov_car,axis=0),inside=trace)
    return saved_poly_fov[(trace[0], trace[1], trace[2])]

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


_saved_rect_tiles_polys = {}
_saved_rect_tiles_centers = {}

def rect_tiles_polys_init(t_ver, t_hor):
    if (t_ver, t_hor) not in _saved_rect_tiles_polys:
        d_hor = degrees_to_radian(360/t_hor)
        d_vert = degrees_to_radian(180/t_ver)
        rect_tile_polys, rect_tile_centers = {}, {}
        for row in range(t_ver):
            rect_tile_polys[row], rect_tile_centers[row] = {}, {}
            for col in range(t_hor):
                theta_c = d_hor * (col + 0.5)
                phi_c = d_vert * (row + 0.5)
                # at eulerian_to_cartesian: theta is hor and phi is ver 
                rect_tile_centers[row][col] = eulerian_to_cartesian(theta_c, phi_c)
                rect_tile_points = points_rect_tile_cartesian(row, col, t_ver, t_hor)
                rect_tile_polys[row][col] = polygon.SphericalPolygon(np.unique(rect_tile_points,axis=0), inside=rect_tile_centers[row][col])
        _saved_rect_tiles_polys[(t_ver, t_hor)] = rect_tile_polys
        _saved_rect_tiles_centers[(t_ver, t_hor)] = rect_tile_centers

def rect_tiles_polys(t_ver, t_hor):
    if (t_ver, t_hor) not in _saved_rect_tiles_polys:
        rect_tiles_polys_init(t_ver, t_hor)
    return _saved_rect_tiles_polys[(t_ver, t_hor)]

def rect_tile_centers(t_ver, t_hor):
    if (t_ver, t_hor) not in _saved_rect_tiles_polys:
        rect_tiles_polys_init(t_ver, t_hor)
    return _saved_rect_tiles_centers[(t_ver, t_hor)]

_vp_55d_rad = degrees_to_radian(110/2)
_vp_110d_rad = degrees_to_radian(110)

class VPExtractTilesRect(VPExtract):
    
    def __init__(self, t_ver, t_hor, cover: VPExtract.Cover):
        self.t_ver, self.t_hor = t_ver, t_hor
        self.shape = (self.t_ver, self.t_hor)
        self.cover = cover
        self.rect_tile_polys = rect_tiles_polys(t_ver, t_hor)
        self.rect_tile_centers = rect_tile_centers(t_ver, t_hor)

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
        fov_poly = poly_fov(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, self.rect_tile_centers[row][col])
                if dist <= _vp_55d_rad:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        rect_tile_poly = self.rect_tile_polys[row][col]
                        view_ratio = rect_tile_poly.overlap(fov_poly)
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly.overlap(rect_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace: NDArray, required_cover: float, return_metrics):
        heatmap = np.zeros((self.t_ver, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly = poly_fov(trace)
        for row in range(self.t_ver):
            for col in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, self.rect_tile_centers[row][col])
                if dist >= _vp_110d_rad:
                    continue
                rect_tile_poly = self.rect_tile_polys[row][col]
                view_ratio = rect_tile_poly.overlap(fov_poly)
                # if view_ratio > 1:
                #     print(f"row={row},col={col},trace={trace}")
                if view_ratio > required_cover:
                    heatmap[row][col] = 1
                    if (return_metrics):
                        areas_out.append(1-view_ratio)
                        vp_quality += fov_poly.overlap(rect_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

saved_voro_tiles_polys = {}

class VPExtractTilesVoro(VPExtract):
    def __init__(self, sphere_voro: SphericalVoronoi, cover: VPExtract.Cover):
        super().__init__()
        self.sphere_voro = sphere_voro
        self.cover = cover
        self.shape = (2, -1)
        if sphere_voro.points.size not in saved_voro_tiles_polys:
            self.voro_tile_polys = {index: polygon.SphericalPolygon(self.sphere_voro.vertices[self.sphere_voro.regions[index]]) for index, _ in enumerate(self.sphere_voro.regions)}
            saved_voro_tiles_polys[sphere_voro.points.size] = self.voro_tile_polys
        else:
            self.voro_tile_polys = saved_voro_tiles_polys[sphere_voro.points.size]
        
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
        fov_poly = poly_fov(trace)
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, _ in enumerate(self.sphere_voro.regions):
            dist = compute_orthodromic_distance(trace, self.sphere_voro.points[index])
            if dist <= _vp_55d_rad:
                heatmap[index] += 1
                if(return_metrics):
                    voro_tile_poly = self.voro_tile_polys[index]
                    view_ratio = voro_tile_poly.overlap(fov_poly)
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly.overlap(voro_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace, required_cover: float, return_metrics):
        areas_out = []
        vp_quality = 0.0
        fov_poly = poly_fov(trace)
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, _ in enumerate(self.sphere_voro.regions):
            voro_tile_poly = self.voro_tile_polys[index]
            view_ratio = voro_tile_poly.overlap(fov_poly)
            if view_ratio > required_cover:
                heatmap[index] += 1
                # view_ratio = 1 if view_ratio > 1 else view_ratio  # fixed with compute_orthodromic_distance
                if(return_metrics):
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly.overlap(voro_tile_poly)
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