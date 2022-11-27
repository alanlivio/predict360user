import unittest

from users360.utils.fov import fov_poly
from users360.utils.tileset import tile_poly

## -- code to generate test_tileset_polys
# t_ver, t_hor = 6 , 4
# traces = [ [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1] ]
# lines = []
# for trace in traces:
#   lines.append(f'# trace = {repr(trace)}\n')
#   for row in range(t_ver): # range(t_ver):
#     for col in range(t_hor):
#       _tile_poly = tile_poly(t_ver, t_hor, row, col)
#       _tile_poly_area = format(_tile_poly.area(),'f')
#       lines.append(f'self.assertAlmostEqual(tile_poly({t_ver}, {t_hor}, {row}, {col}\n  \
# ).area(),{_tile_poly_area},places=6)\n')

#       _fov_poly = fov_poly(trace)
#       _fov_poly_area = format(_fov_poly.area(),'f')
#       lines.append(f'self.assertAlmostEqual(fov_poly({repr(trace)}\n  \
# ).area(),{_fov_poly_area},places=6)\n')

#       tile_poly_overlap = format(_tile_poly.overlap(_fov_poly),'f')
#       lines.append(f'self.assertAlmostEqual(tile_poly({t_ver}, {t_hor}, {row}, {col}\n  \
# ).overlap(fov_poly({repr(trace)})),{tile_poly_overlap},places=6)\n')

#       tile_poly_overlap_negative = format(1-_tile_poly.overlap(_fov_poly),'f')
#       lines.append(f'self.assertAlmostEqual(1-tile_poly({t_ver}, {t_hor}, {row}, {col}\n  \
# ).overlap(fov_poly({repr(trace)})),{tile_poly_overlap_negative},places=6)\n')

# with open('output.py', 'w') as f:
#   f.writelines(lines)


class Test(unittest.TestCase):

  def test_tileset_polys(self) -> None:
    # trace = [1, 0, 0]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([1, 0, 0])),0.999968,places=6)
    # trace = [-1, 0, 0]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([-1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([-1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([-1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([-1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([-1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([-1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([-1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([-1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([-1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([-1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([-1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([-1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([-1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([-1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([-1, 0, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([-1, 0, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([-1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([-1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([-1, 0, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([-1, 0, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([-1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([-1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([-1, 0, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([-1, 0, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([-1, 0, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([-1, 0, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([-1, 0, 0])),1.000000,places=6)
    # trace = [0, 1, 0]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 1, 0])),1.000000,places=6)
    # trace = [0, -1, 0]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, -1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, -1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, -1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, -1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, -1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, -1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, -1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, -1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, -1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, -1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, -1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, -1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, -1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, -1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, -1, 0])),0.618675,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, -1, 0])),0.381325,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, -1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, -1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, -1, 0])),0.433070,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, -1, 0])),0.566930,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, -1, 0])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, -1, 0])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, -1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, -1, 0])),0.999968,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, -1, 0]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, -1, 0])),0.000032,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, -1, 0])),0.999968,places=6)
    # trace = [0, 0, 1]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 0, 1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 0, 1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 0, 1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 0, 1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 0, 1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 0, 1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 0, 1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 0, 1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 0, 1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 0, 1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 0, 1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 0, 1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 0, 1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 0, 1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 0, 1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 0, 1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, 1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 0, 1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 0, 1])),1.000000,places=6)
    # trace = [0, 0, -1]
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 0
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 1
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 2
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 0, 3
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 0
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 1
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 2
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 1, 3
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 0
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 1
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 2
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 2, 3
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 0, -1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 0
      ).overlap(fov_poly([0, 0, -1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 0, -1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 1
      ).overlap(fov_poly([0, 0, -1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 0, -1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 2
      ).overlap(fov_poly([0, 0, -1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).area(),0.927295,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 0, -1])),0.166455,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 3, 3
      ).overlap(fov_poly([0, 0, -1])),0.833545,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 0, -1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 0
      ).overlap(fov_poly([0, 0, -1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 0, -1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 1
      ).overlap(fov_poly([0, 0, -1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 0, -1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 2
      ).overlap(fov_poly([0, 0, -1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).area(),0.500154,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 0, -1])),0.984897,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 4, 3
      ).overlap(fov_poly([0, 0, -1])),0.015103,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 0
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 1
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 2
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).area(),0.143348,places=6)
    self.assertAlmostEqual(fov_poly([0, 0, -1]
      ).area(),3.161202,places=6)
    self.assertAlmostEqual(tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 0, -1])),1.000000,places=6)
    self.assertAlmostEqual( 1-tile_poly(6, 4, 5, 3
      ).overlap(fov_poly([0, 0, -1])),0.000000,places=6)
