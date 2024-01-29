from ecmgen.network import Network
from ecmgen.strandgens import RandomStrandGenerator
from ecmgen.parameters import DomainParameters, RandomStrandGeneratorParameters
from ecmgen.stranddistributions import UniformStrandDistribution
import unittest

from numpy.random import default_rng
import numpy as np


class TestStringMethods(unittest.TestCase):
    def test_creationUniformStrand(self):
        rng = default_rng(1)
        network = Network(DomainParameters(200, 200))
        strand_par = RandomStrandGeneratorParameters(9, 200, 6.25)
        rsg = RandomStrandGenerator(
            strand_par,
            UniformStrandDistribution(network.domain.sizex, network.domain.sizey, rng),
        )
        rsg.build_strands(network)

        self.assertEqual(len(network.beads_positions), 200 * 9)
        self.assertEqual(len(network.beads_types), 200 * 9)
        self.assertEqual(len(network.bonds_groups), 200 * 8)
        self.assertEqual(len(network.bonds_types), 200 * 8)
        self.assertEqual(len(network.angle_groups), 200 * 7)
        self.assertEqual(len(network.angle_types), 200 * 7)

    def test_lenghtOfSingleStrand(self):
        rng = default_rng(1)
        network = Network(DomainParameters(200, 200))
        strand_par = RandomStrandGeneratorParameters(9, 1, 50)
        rsg = RandomStrandGenerator(
            strand_par,
            UniformStrandDistribution(network.domain.sizex, network.domain.sizey, rng),
        )
        rsg.build_strands(network)

        pos = np.array(network.beads_positions)
        number_of_beads = 9 * 1

        self.assertEqual(pos.shape, (number_of_beads, 2))

        norms = np.linalg.norm(pos[:-1] - pos[1:], axis=1)
        self.assertEqual(norms.shape, (8,))
        self.assertEqual(len(norms), 8)

        contour_length = np.sum(norms)
        self.assertAlmostEqual(contour_length, 6.25 * 8)

    def test_directionOfSingleStrand(self):
        rng = default_rng(1)
        network = Network(DomainParameters(200, 200))
        strand_par = RandomStrandGeneratorParameters(9, 1, 50)
        rsg = RandomStrandGenerator(
            strand_par,
            UniformStrandDistribution(network.domain.sizex, network.domain.sizey, rng),
        )
        rsg.build_strands(network)

        pos = np.array(network.beads_positions)
        number_of_beads = 9 * 1

        pos = np.array(network.beads_positions)
        bond_vectors = (pos[:-1, :] - pos[1:, :]) / 6.25
        self.assertEqual(bond_vectors.shape, (8, 2))

        for row in bond_vectors:
            self.assertAlmostEqual(row[0], bond_vectors[0][0])
            self.assertAlmostEqual(row[1], bond_vectors[0][1])


if __name__ == "__main__":
    unittest.main()
