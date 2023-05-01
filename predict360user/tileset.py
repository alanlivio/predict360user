import math
from enum import Enum, auto
from functools import cache

import numpy as np
from scipy.spatial import SphericalVoronoi
from spherical_geometry import polygon
from spherical_geometry.polygon import SphericalPolygon
from tqdm.auto import tqdm

from predict360user import config
from predict360user.utils import *

tqdm.pandas()


@cache
def fov_poly(x, y, z) -> polygon.SphericalPolygon:
  points_trace = fov_points(x, y, z)
  return polygon.SphericalPolygon(points_trace)

@cache
def tile_points(t_ver, t_hor, row, col) -> np.ndarray:
  d_ver = degrees_to_radian(180 / t_ver)
  d_hor = degrees_to_radian(360 / t_hor)
  assert (row < t_ver and col < t_hor)
  points = np.array([
      eulerian_to_cartesian(d_hor * col, d_ver * (row + 1)),
      eulerian_to_cartesian(d_hor * (col + 1), d_ver * (row + 1)),
      eulerian_to_cartesian(d_hor * (col + 1), d_ver * row),
      eulerian_to_cartesian(d_hor * col, d_ver * row),
  ])
  return points


@cache
def _init_tileset(t_ver, t_hor) -> None:
  d_hor = degrees_to_radian(360 / t_hor)
  d_ver = degrees_to_radian(180 / t_ver)
  polys, centers = {}, {}
  for row in range(t_ver):
    polys[row], centers[row] = {}, {}
    for col in range(t_hor):
      points = tile_points(t_ver, t_hor, row, col)
      # sometimes tile points are same on poles
      # they need be unique otherwise it broke the SphericalPolygon.area
      _, idx = np.unique(points, axis=0, return_index=True)
      polys[row][col] = polygon.SphericalPolygon(points[np.sort(idx)])
      theta_c = d_hor * (col + 0.5)
      phi_c = d_ver * (row + 0.5)
      # at eulerian_to_cartesian: theta is hor and phi is ver
      centers[row][col] = eulerian_to_cartesian(theta_c, phi_c)
  return (polys, centers)


def tile_poly(t_ver, t_hor, row, col) -> polygon.SphericalPolygon:
  polys, _ = _init_tileset(t_ver, t_hor)
  return polys[row][col]


def tile_center(t_ver, t_hor, row, col) -> np.array:
  _, centers = _init_tileset(t_ver, t_hor)
  return centers[row][col]


class TileCover(Enum):
  ANY = auto()
  CENTER = auto()
  ONLY20PERC = auto()
  ONLY33PERC = auto()


class TileSet():
  """
  Class for rectangular TileSet
  """

  def __init__(self, t_ver, t_hor, cover=TileCover.ANY) -> None:
    self.t_ver, self.t_hor = t_ver, t_hor
    self.shape = (self.t_ver, self.t_hor)
    self.cover = cover

  cover: TileCover
  shape: tuple[int, int]

  @property
  def title(self) -> str:
    if self.cover == TileCover.ANY:
      return f'{self.prefix}_cov_any'
    elif self.cover == TileCover.CENTER:
      return f'{self.prefix}_cov_ctr'
    elif self.cover == TileCover.ONLY20PERC:
      return f'{self.prefix}_cov_20p'
    elif self.cover == TileCover.ONLY33PERC:
      return f'{self.prefix}_cov_33p'
    else:
      return self.prefix

  @property
  def prefix(self) -> str:
    return f'ts{self.t_ver}x{self.t_hor}'

  def request(self, trace: np.ndarray, return_metrics=False):
    if self.cover == TileCover.CENTER:
      return self._request_110radius_center(trace, return_metrics)
    elif self.cover == TileCover.ANY:
      return self._request_min_cover(trace, 0.0, return_metrics)
    elif self.cover == TileCover.ONLY20PERC:
      return self._request_min_cover(trace, 0.2, return_metrics)
    elif self.cover == TileCover.ONLY33PERC:
      return self._request_min_cover(trace, 0.33, return_metrics)

  def _request_110radius_center(self, trace, return_metrics):
    heatmap = np.zeros((self.t_ver, self.t_hor), dtype=int)
    areas_out = []
    vp_quality = 0.0
    fov_poly_trace = fov_poly(trace[0], trace[1], trace[2])
    for row in range(self.t_ver):
      for col in range(self.t_hor):
        dist = compute_orthodromic_distance(trace, tile_center(self.t_ver, self.t_hor, row, col))
        if dist <= HOR_MARGIN:
          heatmap[row][col] = 1
          if return_metrics:
            try:
              poly_rc = tile_poly(self.t_ver, self.t_hor, row, col)
              view_ratio = poly_rc.overlap(fov_poly_trace)
            except Exception:
              config.error(f'request error for row,col,trace={row},{col},{repr(trace)}')
              continue
            areas_out.append(1 - view_ratio)
            vp_quality += fov_poly_trace.overlap(poly_rc)
    if return_metrics:
      return heatmap, vp_quality, np.sum(areas_out)
    else:
      return heatmap

  def _request_min_cover(self, trace: np.ndarray, required_cover: float, return_metrics):
    heatmap = np.zeros((self.t_ver, self.t_hor), dtype=int)
    areas_out = []
    vp_quality = 0.0
    fov_poly_trace = fov_poly(trace[0], trace[1], trace[2])
    for row in range(self.t_ver):
      for col in range(self.t_hor):
        dist = compute_orthodromic_distance(trace, tile_center(self.t_ver, self.t_hor, row, col))
        if dist >= HOR_DIST:
          continue
        try:
          poly_rc = tile_poly(self.t_ver, self.t_hor, row, col)
          view_ratio = poly_rc.overlap(fov_poly_trace)
        except Exception:
          config.error(f'request error for row,col,trace={row},{col},{repr(trace)}')
          continue
        if view_ratio > required_cover:
          heatmap[row][col] = 1
          if return_metrics:
            areas_out.append(1 - view_ratio)
            vp_quality += fov_poly_trace.overlap(poly_rc)
    if return_metrics:
      return heatmap, vp_quality, np.sum(areas_out)
    else:
      return heatmap


_4X6_CTR = TileSet(4, 6, TileCover.CENTER)
_4X6_ANY = TileSet(4, 6, TileCover.ANY)
_4X6_20P = TileSet(4, 6, TileCover.ONLY20PERC)
TILESET_DEFAULT = _4X6_ANY
TILESET_VARIATIONS = [_4X6_CTR, _4X6_ANY, _4X6_20P]


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
  return {i: SphericalPolygon(voro.vertices[voro.regions[i]]) for i, _ in enumerate(voro.regions)}


@cache
def voro_poly(n_patchs, index) -> SphericalPolygon:
  return _voro_polys(n_patchs)[index]


class TileSetVoro(TileSet):
  """
  Class for Voroni TileSet
  # It uses https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html
  """

  def __init__(self, n_patchs: int, cover=TileCover.ANY) -> None:
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
