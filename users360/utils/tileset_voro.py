import math
from functools import cache

import numpy as np
from scipy.spatial import SphericalVoronoi
from spherical_geometry.polygon import SphericalPolygon

from ..head_motion_prediction.Utils import compute_orthodromic_distance
from .fov import HOR_DIST, HOR_MARGIN, fov_poly
from .tileset import TileCover, TileSet


@cache
def voro_trinity(n_patchs: int) -> SphericalVoronoi:
  points = np.empty((0, 3))
  for index in range(0, n_patchs):
    zi = (1 - 1.0 / n_patchs) * (1 - 2.0 * index / (n_patchs - 1))
    di = math.sqrt(1 - math.pow(zi, 2))
    alphai = index * math.pi * (3 - math.sqrt(5))
    xi = di * math.cos(alphai)
    yi = di * math.sin(alphai)
    new_point = np.array([[xi, yi, zi]])
    points = np.append(points, new_point, axis=0)
  sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))
  sv.sort_vertices_of_regions()
  return sv


@cache
def _voro_polys(n_patchs: int) -> dict:
  voro = voro_trinity(n_patchs)
  return {
      i: SphericalPolygon(voro.vertices[voro.regions[i]])
      for i, _ in enumerate(voro.regions)
  }


@cache
def voro_poly(n_patchs, index) -> SphericalPolygon:
  return _voro_polys(n_patchs)[index]


class TileSetVoro(TileSet):
  """
  Class for Voroni TileSet
  """

  def __init__(self, n_patchs: int, cover = TileCover.ANY) -> None:
    super().__init__(2, -1, cover)  # force shape (2,-1)
    self.n_patchs = n_patchs
    self.voro = voro_trinity(n_patchs)

  @property
  def prefix(self) -> str:
    return f'voro{len(self.voro.points)}'

  def request(self, trace, return_metrics=False):
    if self.cover == TileCover.CENTER:
      return self._request_110radius_center(trace, return_metrics)
    elif self.cover == TileCover.ANY:
      return self._request_min_cover(trace, 0, return_metrics)
    elif self.cover == TileCover.ONLY20PERC:
      return self._request_min_cover(trace, 0.2, return_metrics)
    elif self.cover == TileCover.ONLY33PERC:
      return self._request_min_cover(trace, 0.33, return_metrics)

  def _request_110radius_center(self, trace, return_metrics):
    areas_out = []
    vp_quality = 0.0
    fov_poly_trace = fov_poly(trace[0], trace[1], trace[2])
    heatmap = np.zeros(self.n_patchs)
    for index, _ in enumerate(self.voro.regions):
      dist = compute_orthodromic_distance(trace, self.voro.points[index])
      if dist <= HOR_MARGIN:
        heatmap[index] += 1
        if return_metrics:
          poly = voro_poly(self.n_patchs, index)
          view_ratio = poly.overlap(fov_poly_trace)
          areas_out.append(1 - view_ratio)
          vp_quality += fov_poly_trace.overlap(poly)
    if return_metrics:
      return heatmap, vp_quality, np.sum(areas_out)
    else:
      return heatmap

  def _request_min_cover(self, trace, required_cover: float, return_metrics):
    areas_out = []
    vp_quality = 0.0
    fov_poly_trace = fov_poly(trace[0], trace[1], trace[2])
    heatmap = np.zeros(self.n_patchs)
    for index, _ in enumerate(self.voro.regions):
      dist = compute_orthodromic_distance(trace, self.voro.points[index])
      if dist >= HOR_DIST:
        continue
      poly = voro_poly(self.n_patchs, index)
      view_ratio = poly.overlap(fov_poly_trace)
      if view_ratio > required_cover:
        heatmap[index] += 1
        if return_metrics:
          areas_out.append(1 - view_ratio)
          vp_quality += fov_poly_trace.overlap(poly)
    if return_metrics:
      return heatmap, vp_quality, np.sum(areas_out)
    else:
      return heatmap


_VORO14_CTR = TileSetVoro(14, TileCover.CENTER)
_VORO14_ANY = TileSetVoro(14, TileCover.ANY)
_VORO14_20P = TileSetVoro(14, TileCover.ONLY20PERC)
_VORO24_CTR = TileSetVoro(24, TileCover.CENTER)
_VORO24_ANY = TileSetVoro(24, TileCover.ANY)
_VORO24_20P = TileSetVoro(24, TileCover.ONLY20PERC)

TILESET_VORO_VARIATIONS = [
    _VORO14_CTR, _VORO14_ANY, _VORO14_20P, _VORO24_CTR, _VORO24_ANY, _VORO24_20P
]
TILESET_VORO_DEFAULT = _VORO14_ANY
