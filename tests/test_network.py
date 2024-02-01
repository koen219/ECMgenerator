from ecmgen.networks import random_network, single_strand

import unittest


class TestStringMethods(unittest.TestCase):
    def test_creation(self):
        network = random_network(
            sizex=200,
            sizey=200,
            number_of_beads_per_strand=9,
            number_of_strands=100,
            contour_length_of_strand=50,
            crosslink_max_r=1.0,
            maximal_number_of_initial_crosslinks=50,
            crosslink_bin_size=1 / 3,
            seed=10,
        )

    def test_crosslinks_creation(self):
        network = random_network(
            sizex=200,
            sizey=200,
            number_of_beads_per_strand=9,
            number_of_strands=100,
            contour_length_of_strand=50,
            crosslink_max_r=1.0,
            maximal_number_of_initial_crosslinks=50,
            crosslink_bin_size=1 / 3,
            seed=10,
        )
        for bondtype in network.bonds_types:
            if bondtype == "polymer":
                continue
            self.assertLessEqual(network.details_of_bondtypes[bondtype]["r0"], 1.0)

    def test_singleStrand(self):
        beads = 9
        network = single_strand(
            200,
            200,
            50,50,3.141592653589793,
            beads,
            1*(beads-1),
            None
        )
        print(network.beads_positions) 
        for i in range(beads):
            self.assertEqual(network.beads_positions[i][0], 50+ i)
            self.assertEqual(network.beads_positions[i][1], 50)
        

if __name__ == "__main__":
    unittest.main()
