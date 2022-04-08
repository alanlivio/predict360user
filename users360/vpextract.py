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

LAYOUT = go.Layout(width=600)


def layout_with_title(title):
    return go.Layout(width=800, title=title)


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


def eurelian_sum_to_cartesian(phi_a, theta_a, phi_b, theta_b) -> NDArray:
    # https://math.stackexchange.com/questions/1587910/sum-of-two-vectors-in-spherical-coordinates
    x = np.cos(theta_a) * np.sin(phi_a) + np.cos(theta_b) * np.sin(phi_b)
    y = np.sin(theta_a) * np.sin(phi_a) + np.sin(theta_b) * np.sin(phi_b)
    z = (np.cos(phi_a) + np.cos(phi_b))  # % 1
    return np.array([x, y, z])


def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotated_vector(yaw, pitch, roll, v):
    res = np.matmul(rotation_matrix_z(yaw), np.matmul(rotation_matrix_y(pitch), np.matmul(rotation_matrix_x(roll), v)))
    x, y, z = res[0], res[1], res[2]
    return [x, y, z]


rad360 = degrees_to_radian(360)
rad180 = degrees_to_radian(180)
margin_theta = degrees_to_radian(110/2)
margin_thi = degrees_to_radian(90/2)


def points_fov_cartesian(trace) -> NDArray:
    # https://daglar-cizmeci.com/how-does-virtual-reality-work/#:~:text=Vive%20and%20Rift%20each%20have%20a%20110%2Ddegree%20field%20of%20vision

    # current try (working)
    # theta_vp, phi_vp = cartesian_to_eulerian(*trace)
    # theta_vp_left = theta_vp-margin_theta
    # theta_vp_right = theta_vp+margin_theta
    # phi_vp_bottom = phi_vp-margin_thi
    # phi_vp_top = phi_vp+margin_thi
    # polygon_fov = np.array([
    #     eulerian_to_cartesian(theta_vp_right, phi_vp_bottom),
    #     eulerian_to_cartesian(theta_vp_right, phi_vp_top),
    #     eulerian_to_cartesian(theta_vp_left, phi_vp_top),
    #     eulerian_to_cartesian(theta_vp_left, phi_vp_bottom)
    # ])
    # # current try fix
    # if theta_vp_left < 0 :
    #     theta_vp_left = rad360-abs(theta_vp_left)
    # if theta_vp_right > rad360:
    #     theta_vp_right = theta_vp_right % rad360
    # if phi_vp_bottom < 0:
    #     phi_vp_bottom = rad180-abs(phi_vp_bottom)
    # if phi_vp_top > rad180:
    #     phi_vp_top = phi_vp_top % rad180

    # try sum cartesian
    #  https://math.stackexchange.com/questions/1587910/sum-of-two-vectors-in-spherical-coordinates
    # theta_vp, phi_vp = cartesian_to_eulerian(*trace)
    # theta_vp, phi_vp = eulerian_in_range(theta_vp, phi_vp)
    # margin_theta = degrees_to_radian(110/2)
    # margin_thi = degrees_to_radian(90/2)
    # polygon_fov = np.array([
    #     eurelian_sum_to_cartesian(theta_vp, phi_vp, margin_theta,-margin_thi),
    #     eurelian_sum_to_cartesian(theta_vp, phi_vp, margin_theta, margin_thi),
    #     eurelian_sum_to_cartesian(theta_vp, phi_vp, -margin_theta, margin_thi),
    #     eurelian_sum_to_cartesian(theta_vp, phi_vp, -margin_theta, -margin_thi)
    # ])

    # try rotation from VRClient
    # margin_theta = degrees_to_radian(110/2)
    # margin_thi = degrees_to_radian(90/2)
    # polygon_fov = np.array([
    #     rotated_vector(margin_theta,-margin_thi, 0, trace),
    #     rotated_vector(margin_theta,margin_thi, 0,trace),
    #     rotated_vector(-margin_theta,margin_thi, 0,trace),
    #     rotated_vector( -margin_theta,-margin_thi, 0, trace)
    # ])

    # try https://pypi.org/project/squaternion/
    # # q_margin_theta = Rotation.from_euler('x', 110/2, degrees=True)
    # # q_margin_thi = Rotation.from_euler('x', 90/2, degrees=True)

    # try quarternio 1
    # http://kieranwynn.github.io/pyquaternion/?#from-scalar
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    # theta_vp, phi_vp = cartesian_to_eulerian(*trace)
    # theta_vp, phi_vp = eulerian_in_range(theta_vp, phi_vp)
    # margin_theta = degrees_to_radian(110/2)
    # margin_thi = degrees_to_radian(90/2)
    # polygon_fov = np.array([
    #     Rotation.from_euler('xzy', [1, margin_theta,  -margin_thi]).apply(trace),
    #     Rotation.from_euler('xzy', [1, margin_theta,   margin_thi]).apply(trace),
    #     Rotation.from_euler('xzy', [1, -margin_theta,   margin_thi]).apply(trace),
    #     Rotation.from_euler('xzy', [1, -margin_theta,  -margin_thi]).apply(trace)
    # ])

    #  try quarterion 2
    # polygon_fov_x1y0z0 = np.array([
    #     eulerian_to_cartesian(margin_theta, -margin_thi),
    #     eulerian_to_cartesian(margin_theta, margin_thi),
    #     eulerian_to_cartesian(-margin_theta, margin_thi),
    #     eulerian_to_cartesian(-margin_theta, -margin_thi)
    # ])
    # x1y0z0 = np.array([[1, 0, 0]])
    # trace = np.array([trace])
    # print (x1y0z0.shape)
    # print (trace.shape)
    # rotation, _ = Rotation.align_vectors([x1y0z0], [trace])
    # polygon_fov = np.array([
    #     rotation.apply(polygon_fov_x1y0z0[0]),
    #     rotation.apply(polygon_fov_x1y0z0[1]),
    #     rotation.apply(polygon_fov_x1y0z0[2]),
    #     rotation.apply(polygon_fov_x1y0z0[3]),
    # ])
    # print(polygon_fov_x1y0z0)

    # try quaterion 3 (working)
    margin_theta = degrees_to_radian(90/2)
    margin_thi = degrees_to_radian(110/2)
    polygon_fov_x1y0z0 = np.array([
        eulerian_in_range(margin_theta, -margin_thi),
        eulerian_in_range(margin_theta, margin_thi),
        eulerian_in_range(-margin_theta, margin_thi),
        eulerian_in_range(-margin_theta, -margin_thi)
    ])
    polygon_fov_x1y0z0 = np.array([
        eulerian_to_cartesian(*polygon_fov_x1y0z0[0]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[1]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[2]),
        eulerian_to_cartesian(*polygon_fov_x1y0z0[3])
    ])
    x1y0z0 = np.array([1, 0, 0])
    trace = np.array(trace)
    rotation = rotationBetweenVectors(x1y0z0, trace)
    polygon_fov = np.array([
        rotation.rotate(polygon_fov_x1y0z0[0]),
        rotation.rotate(polygon_fov_x1y0z0[1]),
        rotation.rotate(polygon_fov_x1y0z0[2]),
        rotation.rotate(polygon_fov_x1y0z0[3]),
    ])
    # print(polygon_fov_x1y0z0)
    # print(theta_vp, phi_vp)
    # print(polygon_fov)

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
        self.shape = (len(sphere_voro.points)//6, -1)

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
