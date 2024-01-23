import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .parameters import DomainParameters

BEADTYPE = str
BEADID = int

BOND = Tuple[BEADID, BEADID]
BONDTYPE = str
BONDID = int
FIBREID = int

ANGLE = Tuple[BEADID, BEADID, BEADID]
ANGLETYPE = str


@dataclass
class Network:
    domain: DomainParameters

    beads_positions: List[npt.NDArray[np.float64]] = field(default_factory=list)
    beads_types: List[BEADTYPE] = field(default_factory=list)

    bonds_groups: List[BOND] = field(default_factory=list)
    bonds_types: List[BONDTYPE] = field(default_factory=list)

    angle_groups: List[ANGLE] = field(default_factory=list)
    angle_types: List[ANGLETYPE] = field(default_factory=list)

    # used to store lengths of types
    details_of_bondtypes: Dict[BONDTYPE, Dict[str, float]] = field(default_factory=dict)
