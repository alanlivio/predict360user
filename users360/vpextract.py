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

TILES_H6, TILES_V4 = 6, 4
TRINITY_NPATCHS = 14

def points_voro(npatchs) -> SphericalVoronoi:
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


VORONOI_14P = points_voro(TRINITY_NPATCHS)
VORONOI_24P = points_voro(24)


def points_rect_tile_cartesian(i, j, t_hor, t_vert) -> Tuple[NDArray, NDArray]:
    d_hor = degrees_to_radian(360/t_hor)
    d_vert = degrees_to_radian(180/t_vert)
    theta_c = d_hor * (j + 0.5)
    phi_c = d_vert * (i + 0.5)
    polygon_rect_tile = np.array([
        eulerian_to_cartesian(d_hor * j, d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * i),
        eulerian_to_cartesian(d_hor * j, d_vert * i)])
    return polygon_rect_tile, eulerian_to_cartesian(theta_c, phi_c)


margin_theta = degrees_to_radian(110/2)
margin_thi = degrees_to_radian(90/2)

def points_fov_cartesian(trace) -> NDArray:
    polygon_fov_x1y0z0 = np.array([
        eulerian_in_range(-margin_theta, margin_thi),
        eulerian_in_range(margin_theta, margin_thi),
        eulerian_in_range(margin_theta, -margin_thi),
        eulerian_in_range(-margin_theta, -margin_thi)
    ])
    # polygon_fov_x1y0z0_degrees = np.array([
    #     [radian_to_degrees(polygon_fov_x1y0z0[0][0]),radian_to_degrees(polygon_fov_x1y0z0[0][1])],
    #     [radian_to_degrees(polygon_fov_x1y0z0[1][0]),radian_to_degrees(polygon_fov_x1y0z0[1][1])],
    #     [radian_to_degrees(polygon_fov_x1y0z0[2][0]),radian_to_degrees(polygon_fov_x1y0z0[2][1])],
    #     [radian_to_degrees(polygon_fov_x1y0z0[3][0]),radian_to_degrees(polygon_fov_x1y0z0[3][1])]
    # ])
    # print(polygon_fov_x1y0z0_degrees)
    polygon_fov_x1y0z0 = np.array([
        eulerian_to_cartesian(*polygon_fov_x1y0z0[0]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[1]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[2]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[3])
    ])
    # print(polygon_fov_x1y0z0)
    rotation = rotationBetweenVectors(np.array([1, 0, 0]), np.array(trace))
    polygon_fov = np.array([
        rotation.rotate(polygon_fov_x1y0z0[0]),
        rotation.rotate(polygon_fov_x1y0z0[1]),
        rotation.rotate(polygon_fov_x1y0z0[2]),
        rotation.rotate(polygon_fov_x1y0z0[3]),
    ])
    return polygon_fov


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


class VPExtractTilesRect(VPExtract):
    def __init__(self, t_hor, t_vert, cover: VPExtract.Cover):
        self.t_hor, self.t_vert = t_hor, t_vert
        self.cover = cover
        self.shape = (self.t_vert, self.t_hor)

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
        areas_in = []
        vp_quality = 0.0
        vp_threshold = degrees_to_radian(110/2)
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rect_tile_points, tile_center = points_rect_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = compute_orthodromic_distance(trace, tile_center)
                if dist <= vp_threshold:
                    rect_tile_polygon = polygon.SphericalPolygon(rect_tile_points,)
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(trace))
                    view_area = rect_tile_polygon.overlap(fov_polygon)
                    if view_area > required_cover:
                        heatmap[i][j] = 1
                        if (return_metrics):
                            areas_in.append(view_area)
                            vp_quality += fov_polygon.overlap(rect_tile_polygon)
        return heatmap, vp_quality, np.sum(areas_in)

    def _request_110radius_center(self, trace, return_metrics):
        heatmap = np.zeros((self.t_vert, self.t_hor), dtype=np.int32)
        vp_110_rad_half = degrees_to_radian(110/2)
        areas_in = []
        vp_quality = 0.0
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rect_tile_points, tile_center = points_rect_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = compute_orthodromic_distance(trace, tile_center)
                if dist <= vp_110_rad_half:
                    heatmap[i][j] = 1
                    if (return_metrics):
                        rect_tile_polygon = polygon.SphericalPolygon(rect_tile_points)
                        fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(trace))
                        view_area = rect_tile_polygon.overlap(fov_polygon)
                        areas_in.append(view_area)
                        vp_quality += fov_polygon.overlap(rect_tile_polygon)
        return heatmap, vp_quality, np.sum(areas_in)


class VPExtractTilesVoro(VPExtract):
    def __init__(self, sphere_voro: SphericalVoronoi, cover: VPExtract.Cover):
        self.sphere_voro = sphere_voro
        self.cover = cover
        self.shape = (2, -1)

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
        vp_110_rad_half = degrees_to_radian(110/2)
        areas_in = []
        vp_quality = 0.0
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, region in enumerate(self.sphere_voro.regions):
            dist = compute_orthodromic_distance(trace, self.sphere_voro.points[index])
            if dist <= vp_110_rad_half:
                heatmap[index] += 1
                if(return_metrics):
                    voro_tile_polygon = polygon.SphericalPolygon(self.sphere_voro.vertices[region])
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(trace))
                    view_area = voro_tile_polygon.overlap(fov_polygon)
                    areas_in.append(view_area)
                    vp_quality += fov_polygon.overlap(voro_tile_polygon)
        return heatmap, vp_quality, np.sum(areas_in)

    def _request_min_cover(self, trace, required_cover: float, return_metrics):
        areas_in = []
        vp_quality = 0.0
        heatmap = np.zeros(len(self.sphere_voro.regions))
        for index, region in enumerate(self.sphere_voro.regions):
            voro_tile_polygon = polygon.SphericalPolygon(self.sphere_voro.vertices[region])
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(trace))
            view_area = voro_tile_polygon.overlap(fov_polygon)
            if view_area > required_cover:
                heatmap[index] += 1
                # view_area = 1 if view_area > 1 else view_area  # fixed with compute_orthodromic_distance
                if(return_metrics):
                    areas_in.append(view_area)
                    vp_quality += fov_polygon.overlap(voro_tile_polygon)
        return heatmap, vp_quality, np.sum(areas_in)


VPEXTRACT_VORO_14_CENTER = VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.CENTER)
VPEXTRACT_RECT_6_4_CENTER = VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER)
VPEXTRACT_RECT_8_6_CENTER = VPExtractTilesRect(8, 6, VPExtract.Cover.CENTER)

VPEXTRACTS_VORO = [
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.CENTER),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ANY),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ONLY20PERC),
    # VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ONLY33PERC),
    # VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.CENTER),
    # VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ANY),
    # VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ONLY33PERC),
    # VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ONLY20PERC),
]
VPEXTRACTS_RECT = [
    VPExtractTilesRect(6, 4, VPExtract.Cover.ANY),
    VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER),
    VPExtractTilesRect(6, 4, VPExtract.Cover.ONLY20PERC),
    # VPExtractTilesRect(6, 4, VPExtract.Cover.ONLY33PERC),
]
VPEXTRACT_METHODS = [*VPEXTRACTS_VORO, *VPEXTRACTS_RECT]
