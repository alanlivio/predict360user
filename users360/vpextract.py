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
    for i in range(0, npatchs):
        zi = (1 - 1.0/npatchs) * (1 - 2.0*i / (npatchs - 1))
        di = math.sqrt(1 - math.pow(zi, 2))
        alphai = i * math.pi * (3 - math.sqrt(5))
        xi = di * math.cos(alphai)
        yi = di * math.sin(alphai)
        new_point = np.array([[xi, yi, zi]])
        points = np.append(points, new_point, axis=0)
    sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    return sv
VORONOI_14P = create_trinity_voro(14)
VORONOI_24P = create_trinity_voro(24)

def points_rect_tile_cartesian(i, j, t_hor, t_vert) -> NDArray:
    d_hor = degrees_to_radian(360/t_hor)
    d_vert = degrees_to_radian(180/t_vert)
    polygon_rect_tile = np.array([
        eulerian_to_cartesian(d_hor * j, d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * i),
        eulerian_to_cartesian(d_hor * j, d_vert * i)])
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
X1Y0Z0_POLYG_FOV_AREA = polygon.SphericalPolygon(X1Y0Z0_POLYG_FOV).area()

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
        saved_poly_fov[(trace[0], trace[1], trace[2])] = polygon.SphericalPolygon(points_fov_cartesian(trace))
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

def rect_tiles_polys_init(t_vert, t_hor):
    if (t_vert, t_hor) not in _saved_rect_tiles_polys:
        d_hor = degrees_to_radian(360/t_hor)
        d_vert = degrees_to_radian(180/t_vert)
        rect_tile_polys, rect_tile_centers = {}, {}
        for i in range(t_vert):
            rect_tile_polys[i], rect_tile_centers[i] = {}, {}
            for j in range(t_hor):
                theta_c = d_hor * (j + 0.5)
                phi_c = d_vert * (i + 0.5)
                rect_tile_points = points_rect_tile_cartesian(i, j, t_hor, t_vert)
                rect_tile_polys[i][j] = polygon.SphericalPolygon(rect_tile_points)
                rect_tile_centers[i][j] = eulerian_to_cartesian(theta_c, phi_c)
        _saved_rect_tiles_polys[(t_vert, t_hor)] = rect_tile_polys
        _saved_rect_tiles_centers[(t_vert, t_hor)] = rect_tile_centers

def rect_tiles_polys(t_vert, t_hor):
    if (t_vert, t_hor) not in _saved_rect_tiles_polys:
        rect_tiles_polys_init(t_vert, t_hor)
    return _saved_rect_tiles_polys[(t_vert, t_hor)]

def rect_tile_centers(t_vert, t_hor):
    if (t_vert, t_hor) not in _saved_rect_tiles_polys:
        rect_tiles_polys_init(t_vert, t_hor)
    return _saved_rect_tiles_centers[(t_vert, t_hor)]

_vp_55d_rad = degrees_to_radian(110/2)

class VPExtractTilesRect(VPExtract):
    
    def __init__(self, t_hor, t_vert, cover: VPExtract.Cover):
        self.t_hor, self.t_vert = t_hor, t_vert
        self.shape = (self.t_vert, self.t_hor)
        self.cover = cover
        self.rect_tile_polys = rect_tiles_polys(t_vert, t_hor)
        self.rect_tile_centers = rect_tile_centers(t_vert, t_hor)

    @property
    def title(self):
        prefix = f'vpextract_rect{self.t_hor}x{self.t_vert}'
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

    def _request_min_cover(self, trace: NDArray, required_cover: float, return_metrics):
        heatmap = np.zeros((self.t_vert, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly = poly_fov(trace)
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, self.rect_tile_centers[i][j])
                if dist <= _vp_55d_rad:
                    rect_tile_poly = self.rect_tile_polys[i][j]
                    view_ratio = rect_tile_poly.overlap(fov_poly)
                    if view_ratio > required_cover:
                        heatmap[i][j] = 1
                        if (return_metrics):
                            areas_out.append(1-view_ratio)
                            vp_quality += fov_poly.overlap(rect_tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_110radius_center(self, trace, return_metrics):
        heatmap = np.zeros((self.t_vert, self.t_hor), dtype=np.int32)
        areas_out = []
        vp_quality = 0.0
        fov_poly = poly_fov(trace)
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                dist = compute_orthodromic_distance(trace, self.rect_tile_centers[i][j])
                if dist <= _vp_55d_rad:
                    heatmap[i][j] = 1
                    if (return_metrics):
                        rect_tile_poly = self.rect_tile_polys[i][j]
                        view_ratio = rect_tile_poly.overlap(fov_poly)
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
            self.voro_tile_polys = {i: polygon.SphericalPolygon(self.sphere_voro.vertices[self.sphere_voro.regions[i]]) 
                    for i, _ in enumerate(self.sphere_voro.regions)}
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


VPEXTRACT_RECT_6_4_CENTER = VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER)
VPEXTRACT_RECT_6_4_ANY = VPExtractTilesRect(6, 4, VPExtract.Cover.ANY)
VPEXTRACT_RECT_6_4_20PERC = VPExtractTilesRect(6, 4, VPExtract.Cover.ONLY20PERC)
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
    VPEXTRACT_RECT_6_4_CENTER,
    VPEXTRACT_RECT_6_4_ANY,
    VPEXTRACT_RECT_6_4_20PERC,
]
VPEXTRACT_METHODS = [*VPEXTRACTS_VORO, *VPEXTRACTS_RECT]