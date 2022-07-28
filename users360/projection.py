import numpy as np
from .tileset import *
from .tileset_voro import *
from head_motion_prediction.Utils import *
from scipy.spatial import SphericalVoronoi, geometric_slerp
import plotly.graph_objs as go


class Projection():

    def __init__(self, tileset=TileSet.default()):
        if isinstance(tileset, TileSetVoro):
            self.data = self._init_data_sphere_voro(tileset.voro)
        else:
            self.data = self._init_data_sphere_tiles(tileset.t_ver, tileset.t_hor)
        self.title = f"project"

    def _init_data_sphere(self) -> list:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(theta), np.sin(phi)) * 0.98
        y = np.outer(np.sin(theta), np.sin(phi)) * 0.98
        z = np.outer(np.ones(100), np.cos(phi)) * 0.98
        # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
        colorscale = [[0, "rgba(200, 0, 0, 0.1)"], [1.0, "rgba(255, 0, 0, 0.1)"]]
        return [go.Surface(x=x, y=y, z=z, colorscale=colorscale, showlegend=False, showscale=False)]

    def _init_data_sphere_voro(self, sphere_voro: SphericalVoronoi, with_generators=False) -> list:
        data = self._init_data_sphere()
        # generator points
        if with_generators:
            gens = go.Scatter3d(x=sphere_voro.points[:, 0],
                                y=sphere_voro.points[:, 1],
                                z=sphere_voro.points[:, 2],
                                mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='voron center')
            data.append(gens)
        # edges
        for region in sphere_voro.regions:
            n = len(region)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = sphere_voro.vertices[region][index]
                end = sphere_voro.vertices[region][(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2],
                                    mode='lines', line={'width': 2, 'color': 'gray'}, showlegend=False)
                data.append(edge)
        return data

    def _init_data_sphere_tiles(self, t_ver, t_hor) -> list:
        data = self._init_data_sphere()
        for row in range(t_ver):
            for col in range(t_hor):
                # -- add tiles edges
                tile_points = TileSet.tile_points(t_ver, t_hor, row, col)
                n = len(tile_points)
                t = np.linspace(0, 1, 100)
                for index in range(n):
                    start = tile_points[index]
                    end = tile_points[(index + 1) % n]
                    result = np.array(geometric_slerp(start, end, t))
                    edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                        'width': 2, 'color': 'gray'}, name='region edge', showlegend=False)
                    data.append(edge)
        return data

    def add_polygon(self, polygon: polygon.SphericalPolygon):
        self._add_points([point for point in polygon.points])  # polygon.points is a generator

    def add_polygon_as_points(self, points):
        assert points.shape[1] == 3  # check if cartesian
        self._add_points(points)

    def add_polygon_as_row_col_tile(self, t_ver, t_hor, row, col):
        points = TileSet.tile_points(t_ver, t_hor, row, col)
        self._add_points(points)

    def add_polygon_from_trace(self, trace):
        points = FOV.points(trace)
        self._add_points(points)

    def add_trace(self, trace):
        self.data.append(go.Scatter3d(x=[trace[0]], y=[trace[1]], z=[trace[2]],
                                      mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': 'red'}))

    def add_vp(self, trace):
        points_fov = FOV.points(trace)
        n = len(points_fov)
        gens = go.Scatter3d(x=points_fov[:, 0],
                            y=points_fov[:, 1],
                            z=points_fov[:, 2],
                            mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points_fov))]}, name='fov corner', showlegend=False)
        self.data.append(gens)
        t = np.linspace(0, 1, 100)
        for index in range(n):
            start = points_fov[index]
            end = points_fov[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            self.data.append(edge)

    def _add_points(self, points):
        # points
        n = len(points)
        t = np.linspace(0, 1, 100)
        gens = go.Scatter3d(x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode='markers', showlegend=False, marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points))]})
        self.data.append(gens)
        for index in range(n):
            start = points[index]
            end = points[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            self.data.append(edge)

    def show(self):
        fig = go.Figure(data=self.data)
        fig.update_layout(width=800, showlegend=False, title_text=self.title)
        fig.show()
