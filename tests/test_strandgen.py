from ecmgen.strandgens import RandomStrandGenerator
from ecmgen.parameters import Parameter
from ecmgen.stranddistributions import UniformStrandDistribution
import unittest

from numpy.random import default_rng
import numpy as np

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

    def test_lenghtOfSingleStrand(self):
        rng = default_rng(1)
        par = Parameter(
            sizex = 200,
            sizey = 200,
            number_of_beads_per_strand = 9,
            number_of_strands = 1,
            contour_length_of_strand=6.25*8
        )
        rsg = RandomStrandGenerator(
            UniformStrandDistribution(rng, par.Lx, par.Ly)
        )
        net = rsg.build_strands(par)
        
        pos = np.array(net.beads_positions)
        
        norms = np.linalg.norm(pos[:-1] - pos[1:], axis=1)
        self.assertEqual(len(norms) , 8 )
        self.assertEqual(norms.shape, (8,))
        
        contour_length = np.sum(norms) 
        self.assertAlmostEqual(contour_length, 6.25*8)
        
    def test_directionOfSingleStrand(self):
        rng = default_rng(1)
        par = Parameter(
            sizex = 200,
            sizey = 200,
            number_of_beads_per_strand = 9,
            number_of_strands = 1,
            contour_length_of_strand=6.25*8
        )
        rsg = RandomStrandGenerator(
            UniformStrandDistribution(rng, par.Lx, par.Ly)
        )
        net = rsg.build_strands(par)
        
        pos = np.array(net.beads_positions)
        bond_vectors = (pos[:-1,:]-pos[1:,:])/ 6.25
        self.assertEqual(bond_vectors.shape, (8,2))
       
        for row in bond_vectors:
            self.assertAlmostEqual(row[0], bond_vectors[0][0]) 
            self.assertAlmostEqual(row[1], bond_vectors[0][1]) 
      

if __name__ == '__main__':
    unittest.main()