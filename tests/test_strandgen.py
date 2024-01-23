from .strandgens import RandomStrandGenerator
from .parameters import Parameter
from .stranddistributions import UniformStrandDistribution
import unittest

from numpy.random import default_rng

class TestStringMethods(unittest.TestCase):

    def test_creationUniformStrand(self):
        rng = default_rng(1)
        par = Parameter(
            200,
            200,
            9,
            200,
            6.25 
        )
        rsg = RandomStrandGenerator(
            UniformStrandDistribution(rng, par.Lx, par.Ly)
        )
        net = rsg.build_strands(par)
        
        self.assertEqual(len(net.beads_positions) , 200 * 9 )
        self.assertEqual(len(net.beads_types) , 200 * 9 )
        self.assertEqual(len(net.bonds_groups), 200 * 8)
        self.assertEqual(len(net.bonds_types), 200 * 8)
        self.assertEqual(len(net.angle_groups), 200 * 7)
        self.assertEqual(len(net.angle_types), 200 * 7)


if __name__ == '__main__':
    unittest.main()