import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import List, Tuple

BEADTYPE = str
BEADID = int

BOND = Tuple[BEADID, BEADID]
BONDTYPE = str

ANGLE = Tuple[BEADID, BEADID, BEADID]
ANGLETYPE = str


@dataclass
class Network:
    beads_positions: List[npt.NDArray[np.float64]] = field(default_factory=list)
    beads_types: List[BEADTYPE] = field(default_factory=list)

    bonds_groups: List[BOND] = field(default_factory=list)
    bonds_types: List[BONDTYPE] = field(default_factory=list)

    angle_groups: List[ANGLE] = field(default_factory=list)
    angle_types: List[ANGLETYPE] = field(default_factory=list)
