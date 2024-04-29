from .stranddistributions import (
    UniformStrandDistribution,
    DeterministicStrandDistribution,
    StrandDistributionGeneral,
    VonMisesStrandDistribution,
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


def random_directed_network(
    sizex,
    sizey,
    number_of_beads_per_strand,
    number_of_strands,
    direction_spread,
    direction_angle,
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
            VonMisesStrandDistribution(
                sizex, sizey, direction_angle, direction_spread, seed
            ),
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

def ISV_network(
    sizex,
    sizey,
    number_of_beads_per_strand,
    number_of_strands,
    contour_length_of_strand,
    crosslink_max_r,
    maximal_number_of_initial_crosslinks,
    crosslink_bin_size,
    spread_xaxis=1.0,
    seed=None,
    fix_boundary=None,
) -> Network:
    rng = np.random.default_rng(seed)
    if fix_boundary:
        domain = DomainParameters(sizex, sizey, fix_boundary=fix_boundary)
    else:
        domain = DomainParameters(
            sizex, sizey, fix_boundary_north=True, fix_boundary_south=True
        )
    nt = NetworkType(
        domain,
        RandomStrandGenerator(
            RandomStrandGeneratorParameters(
                number_of_beads_per_strand=number_of_beads_per_strand,
                number_of_strands=number_of_strands,
                contour_length_of_strand=contour_length_of_strand,
            ),
            StrandDistributionGeneral(
                lambda n: rng.normal(sizex * 0.5, spread_xaxis, n),
                lambda n: rng.uniform(0, sizey, n),
                lambda n: rng.uniform(0, 2 * np.pi, n),
            ),
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

def laminin(sizex,sizey, amount_of_laminin, network: Network, seed=None):
    assert len(network.beads_positions) >= amount_of_laminin
    rng = np.random.default_rng(seed) 
    free_indices = [k for k, typ in enumerate(network.beads_types) if typ == 'free']
    free_beads_that_get_laminin_connection = list(rng.choice(
        free_indices,
        size=amount_of_laminin,
        replace=False
    ))
    laminin_positions = [network.beads_positions[k] for k in free_beads_that_get_laminin_connection]
    laminin_types = ['boundary'] * amount_of_laminin
    laminin_ids = [len(network.beads_positions) + k for k in range(amount_of_laminin)]

    laminin_bonds = list(zip(
        free_beads_that_get_laminin_connection,
        laminin_ids
    ))

    laminin_bonds_types = ['laminin'] * amount_of_laminin

    network.details_of_bondtypes['laminin'] = {
        'k': 1.0,# ratio of spring_k,
        'r0': 0.0 # used
    }

    network.beads_positions.extend(laminin_positions)
    network.beads_types.extend(laminin_types)

    network.bonds_groups.extend(laminin_bonds)
    network.bonds_types.extend(laminin_bonds_types)



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


from .regular_network import (
    RegularNetwork,
    RegularNetworkParameters,
    RegularCrosslinker,
)


def regular(
    sizex,
    sizey,
    number_of_fibers_per_side,
    number_of_beads_per_strand,
    fix_boundary,
    single_side=False
):
    print("single_side = ", single_side)
    par = RegularNetworkParameters(
        number_of_fibers_per_side * 2, number_of_beads_per_strand,only_vertical_strands=single_side
    )

    cross = RegularCrosslinker(par) if not single_side else None

    nt = NetworkType(
        DomainParameters(sizex, sizey, fix_boundary),
        RegularNetwork(par),
        cross,
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
