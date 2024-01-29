
import numpy as np
import numpy.typing as npt
from typing import Optional
from .strandgens import StrandGenerator
from .parameters import DomainParameters
from .crosslink_distributors import CrosslinkDistributer

from .network import Network

class NetworkType:
    """Config class of the network. Used to generate a Network class.
    """
    
    def __init__(
        self,
        domain: DomainParameters,
        strandgenerator: StrandGenerator,
        crosslink_distributor: Optional[CrosslinkDistributer],
        seed: Optional[int] = None
    ):
        self._strand_generator: StrandGenerator = strandgenerator
        self._crosslink_distributor = crosslink_distributor
        
        self._rng = np.random.default_rng(seed=seed)
        
        self._network = Network(domain)

    def generate(self) -> Network:
        """Generates a network from the generators. Throws exceptions when some network are not neatly generated.
        """
        self._strand_generator.build_strands(self._network) 
        self._strand_generator.fix_boundaries(self._network)
        if self._crosslink_distributor:
            self._crosslink_distributor.distribute_crosslinkers(self._network)
        return self._network
    
    @property
    def network(self):
        return self._network
