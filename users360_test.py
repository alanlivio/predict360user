from users360 import *
import unittest
from numpy import ndarray


class User360Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.one_user = Plot(Dataset().get_traces_one_video_one_user(), title_sufix="one_video_one_user")
        cls.all_users = Plot(Dataset().get_traces_one_video_all_users(), title_sufix="one_video_all_users")
        cls.some_users = Plot(Dataset().get_traces_random_one_user(4), title_sufix="some_users")
        cls.one_trace = Plot(Dataset().get_one_trace(), title_sufix="1_trace")

    def test_plot_traces_shape(self):
        self.assertIsNotNone(self.one_trace)
        self.assertIsInstance(self.one_user.traces, ndarray)
        self.assertIsNotNone(self.all_users)
        self.assertIsInstance(self.all_users.traces, ndarray)
        self.assertIsNotNone(self.some_users)
        self.assertIsInstance(self.some_users.traces, ndarray)
        self.assertIsNotNone(self.one_trace)
        self.assertIsInstance(self.one_trace.traces, ndarray)

    def test_plot_sphere_rectan(self):
        self.one_user.sphere_rectan(6, 4, to_html=True)
        self.one_user.sphere_rectan(8, 6, to_html=True)
        self.one_trace.sphere_rectan_with_vp(6, 4, to_html=True)
        self.one_trace.sphere_rectan_with_vp(8, 6, to_html=True)
        self.some_users.metrics_vpextract(VPEXTRACS_RECT, plot_bars=False,
                                               plot_traces=False, plot_heatmaps=False)

    def test_plot_shpere_voro(self):
        self.one_user.sphere_voro(VORONOI_14P, to_html=True)
        self.one_user.sphere_voro(VORONOI_24P, to_html=True)
        self.one_trace.sphere_voro_with_vp(VORONOI_24P, to_html=True)
        self.one_trace.sphere_voro_with_vp(VORONOI_24P, to_html=True)
        self.some_users.metrics_vpextract(VPEXTRACS_VORO, plot_bars=False,
                                               plot_traces=False, plot_heatmaps=False)

    def test_plot_erp(self):
        self.one_user.erp_heatmap(VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER), to_html=True)

    def test_entropy(self):
        users_entropy = Dataset().get_cluster_entropy_by_vpextract()
        self.assertIsNotNone(users_entropy)


if __name__ == '__main__':
    unittest.main()
