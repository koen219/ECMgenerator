from .stranddistributions import (
    UniformStrandDistribution,
    DeterministicStrandDistribution,
)
from .strandgens import RandomStrandGenerator
from .crosslink_distributors import TipToTailCrosslinkDistributer
from .density_crosslinker import StrandDensityCrosslinkDistributer
from .parameters import (
    DomainParameters,
    RandomStrandGeneratorParameters,
    StrandDensityCrosslinkDistributerParameters,
)
from .density_crosslinker import StrandDensityCrosslinkDistributer
from .parameters import (
    DomainParameters,
    RandomStrandGeneratorParameters,
    StrandDensityCrosslinkDistributerParameters,
)
from .networktype import NetworkType
from .network import Network
import numpy.random as npr
import numpy as np


def random_network(
    sizex,
    sizey,
    number_of_beads_per_strand,
    number_of_strands,
    contour_length_of_strand,
    crosslink_max_r,
    maximal_number_of_initial_crosslinks,
    crosslink_bin_size,
    seed=None,
    fix_boundary=False,
) -> Network:
    nt = NetworkType(
        DomainParameters(sizex, sizey, fix_boundary=fix_boundary),
        RandomStrandGenerator(
            RandomStrandGeneratorParameters(
                number_of_beads_per_strand=number_of_beads_per_strand,
                number_of_strands=number_of_strands,
                contour_length_of_strand=contour_length_of_strand,
            ),
            UniformStrandDistribution(sizex, sizey, seed),
        ),
        StrandDensityCrosslinkDistributer(
            StrandDensityCrosslinkDistributerParameters(
                crosslink_max_r,
                maximal_number_of_initial_crosslinks,
                number_of_beads_per_strand,
                number_of_strands,
                crosslink_bin_size,
            ),
            seed=seed,
        ),
        seed=seed,
    )
    return nt.generate()


def single_strand(
    sizex,
    sizey,
    start_x,
    start_y,
    angle,
    number_of_beads_per_strand,
    contour_length_of_strand,
    seed=None,
) -> Network:
    beads_to_middle_of_strand = number_of_beads_per_strand // 2
    length_single_bond = contour_length_of_strand / (number_of_beads_per_strand - 1)

    middle_of_strand_x = (
        start_x - np.cos(angle) * beads_to_middle_of_strand * length_single_bond
    )
    middle_of_strand_y = (
        start_y - np.sin(angle) * beads_to_middle_of_strand * length_single_bond
    )

    nt = NetworkType(
        DomainParameters(sizex, sizey, fix_boundary=True),
        RandomStrandGenerator(
            RandomStrandGeneratorParameters(
                number_of_beads_per_strand=number_of_beads_per_strand,
                number_of_strands=1,
                contour_length_of_strand=contour_length_of_strand,
            ),
            DeterministicStrandDistribution(
                [middle_of_strand_x], [middle_of_strand_y], [angle]
            ),
        ),
        None,
        seed=seed,
    )
    return nt.generate()


def single_spring(
    sizex,
    sizey,
    number_of_strands,
    number_of_beads_per_strand,
    contour_length_of_strand,
    seed=None,
) -> Network:

    assert number_of_strands % 2 == 1
    n = (number_of_strands - 1) // 2
    x_pos = [100 + 100.0 * k / (2.0 * n) for k in range(number_of_strands)]
    angle_up = np.arctan(np.sqrt(contour_length_of_strand**2 - (1 / n) ** 2) * 2 * n)
    angle_down = np.arctan(-np.sqrt(contour_length_of_strand**2 - (1 / n) ** 2) * 2 * n)
    angles = [angle_up, angle_down] * ((number_of_strands - 1) // 2) + [angle_up]

    nt = NetworkType(
        DomainParameters(sizex, sizey, fix_boundary=True),
        RandomStrandGenerator(
            RandomStrandGeneratorParameters(
                number_of_beads_per_strand=number_of_beads_per_strand,
                number_of_strands=number_of_strands,
                contour_length_of_strand=contour_length_of_strand,
            ),
            DeterministicStrandDistribution(x_pos, [100] * number_of_strands, angles),
        ),
        TipToTailCrosslinkDistributer(
            number_of_beads_per_strand=number_of_beads_per_strand,
            number_of_strands=number_of_strands,
        ),
        seed=seed,
    )
    return nt.generate()
