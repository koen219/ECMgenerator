
import numpy as np
import numpy.typing as npt
from typing import Optional

from ecmgen.network import Network


class NetworkType:
    """Config class of the network. Used to generate a Network class.
    """
    
    def __init__(
        self,
        strandgenerator,
        crosslinker,
        seed: Optional[int] = None
    ):
        self._strandgeneator = strandgenerator
        self._crosslinker = crosslinker
        
        self._rng = np.random.default_rng(seed=seed)

    def generate(self, beads_per_strand: Optional[int], strands: Optional[int], max_crosslinkers: Optional[int]) -> Network:
        """Generates a network from the generators. Throws exceptions when some network are not neatly generated.

        Args:
            beads_per_strand (Optional[int]): Beads per strand, put None if required by choosen strandgenerator
            strands (Optional[int]): Number of strands, put None if required by choosen strandgenerator
            max_crosslinkers (Optional[int]): Maximal number of crosslinker, if the crosslinker recruires this.
        """
        raise NotImplementedError()
