from head_motion_prediction.Utils import *
from scipy.spatial import SphericalVoronoi
from spherical_geometry import polygon
import numpy as np
import math
import plotly.express as px
from .tiles import *

def voro_trinity(npatchs) -> SphericalVoronoi:
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
VORONOI_14P = voro_trinity(14)
VORONOI_24P = voro_trinity(24)


class TilesVoro(TilesIF):
    _saved_polys = None
    _default = None
    _variations = None
    
    def __init__(self, voro: SphericalVoronoi, cover: TileCover):
        super().__init__()
        self.voro = voro
        self.cover = cover
        self.shape = (2, -1)
    
    @classmethod
    def tile_poly(cls, voro: SphericalVoronoi, index):
        if cls._saved_polys is None:
            cls._saved_polys = {}
        if voro.points.size not in cls._saved_polys:
            polys = {i: polygon.SphericalPolygon(voro.vertices[voro.regions[i]]) for i, _ in enumerate(voro.regions)}
            cls._saved_polys[voro.points.size] = polys
        return cls._saved_polys[voro.points.size][index]

    @classmethod
    def default(cls):
        if cls._default is None:
            cls._default = TilesVoro(VORONOI_14P, TileCover.CENTER)
        return cls._default
    
    @classmethod
    def variations(cls):
        if cls._variations is None:
            cls._variations = [
            TilesVoro(VORONOI_14P, TileCover.CENTER),
            TilesVoro(VORONOI_14P, TileCover.ANY),
            TilesVoro(VORONOI_14P, TileCover.ONLY20PERC),
            # TilesVoro(VORONOI_24P, TileCover.CEMTER),
            # TilesVoro(VORONOI_24P, TileCover.ANY),
            # TilesVoro(VORONOI_24P, TileCover.ONLY20PERC),
        ]
        return cls._variations
    
    @property
    def title(self):
        prefix = f'voro{len(self.voro.points)}'
        match self.cover:
            case TileCover.ANY:
                return f'{prefix}_any'
            case TileCover.CENTER:
                return f'{prefix}_center'
            case TileCover.ONLY20PERC:
                return f'{prefix}_20prt'
            case TileCover.ONLY33PERC:
                return f'{prefix}_33prt'

    def request(self, trace, return_metrics=False):
        match self.cover:
            case TileCover.CENTER:
                return self._request_110radius_center(trace, return_metrics)
            case TileCover.ANY:
                return self._request_min_cover(trace, 0, return_metrics)
            case TileCover.ONLY20PERC:
                return self._request_min_cover(trace, 0.2, return_metrics)
            case TileCover.ONLY33PERC:
                return self._request_min_cover(trace, 0.33, return_metrics)

    def _request_110radius_center(self, trace, return_metrics):
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = FOV.poly(trace)
        heatmap = np.zeros(len(self.voro.regions))
        for index, _ in enumerate(self.voro.regions):
            dist = compute_orthodromic_distance(trace, self.voro.points[index])
            if dist <= FOV.HOR_MARGIN:
                heatmap[index] += 1
                if(return_metrics):
                    tile_poly = TilesVoro.tile_poly(self.voro, index)
                    view_ratio = tile_poly.overlap(fov_poly_trace)
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly_trace.overlap(tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)

    def _request_min_cover(self, trace, required_cover: float, return_metrics):
        areas_out = []
        vp_quality = 0.0
        fov_poly_trace = FOV.poly(trace)
        heatmap = np.zeros(len(self.voro.regions))
        for index, _ in enumerate(self.voro.regions):
            dist = compute_orthodromic_distance(trace, self.voro.points[index])
            if dist >= FOV.HOR_DIST:
                continue    
            tile_poly = TilesVoro.tile_poly(self.voro,index)
            view_ratio = tile_poly.overlap(fov_poly_trace)
            if view_ratio > required_cover:
                heatmap[index] += 1
                if(return_metrics):
                    areas_out.append(1-view_ratio)
                    vp_quality += fov_poly_trace.overlap(tile_poly)
        return heatmap, vp_quality, np.sum(areas_out)
