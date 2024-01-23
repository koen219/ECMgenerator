from .stranddistributions import UniformStrandDistribution
from .strandgens import RandomStrandGenerator
from .parameters import Parameter

import numpy.random as npr

def random_network(par):
    rng = npr.default_rng(0)
    rsg = RandomStrandGenerator(
            UniformStrandDistribution(rng, par.Lx, par.Ly)
        )
    net = rsg.build_strands(par)
    return net
