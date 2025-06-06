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
    """
    Represents a network of beads, bonds, and angles with associated properties.
    """

    domain: DomainParameters

    beads_positions: List[npt.NDArray[np.float64]] = field(default_factory=list)
    beads_types: List[BEADTYPE] = field(default_factory=list)

    bonds_groups: List[BOND] = field(default_factory=list)
    bonds_types: List[BONDTYPE] = field(default_factory=list)

    angle_groups: List[ANGLE] = field(default_factory=list)
    angle_types: List[ANGLETYPE] = field(default_factory=list)

    # used to store lengths of types
    details_of_bondtypes: Dict[BONDTYPE, Dict[str, float]] = field(default_factory=dict)

    # used to store lengths of types
    details_of_angletypes: Dict[ANGLETYPE, Dict[str, float]] = field(
        default_factory=dict
    )

    def __radd__(self, other):
        """Implementation of this function allows the use of Network in python 'sum'."""
        if other == 0:
            return self
        return self.__add__(other)

    def __add__(self, other):
        """
        Combines two networks by merging their beads, bonds, angles.
        """
        net = Network(self.domain)

        net.beads_positions = [x.copy() for x in self.beads_positions]
        net.beads_types = [x for x in self.beads_types]

        net.bonds_groups = [x for x in self.bonds_groups]
        net.bonds_types = [x for x in self.bonds_types]

        net.angle_groups = [x for x in self.angle_groups]
        net.angle_types = [x for x in self.angle_types]

        bead_id_offset = len(net.beads_positions)

        net.beads_positions += [x.copy() for x in other.beads_positions]
        net.beads_types += [x for x in other.beads_types]

        net.bonds_groups += [
            list(map(lambda b: b + bead_id_offset, x)) for x in other.bonds_groups
        ]
        net.bonds_types += [x for x in other.bonds_types]

        net.angle_groups += [
            list(map(lambda b: b + bead_id_offset, x)) for x in other.angle_groups
        ]
        net.angle_types += [x for x in other.angle_types]

        new_details = dict()
        for key, value in self.details_of_bondtypes.items():
            print(key, value)
            print(other.details_of_bondtypes)
            if (
                key in other.details_of_bondtypes.keys()
                and value["k"] != other.details_of_bondtypes[key]["k"]
                and value["r0"] != other.details_of_bondtypes[key]["r0"]
            ):
                raise RuntimeError(
                    "Problem adding networks. Both have specified bond details but they are not the same!!"
                )
            new_details[key] = value

        for key, value in other.details_of_bondtypes.items():
            if (
                key in self.details_of_bondtypes.keys()
                and value["k"] != self.details_of_bondtypes[key]["k"]
                and value["r0"] != self.details_of_bondtypes[key]["r0"]
            ):
                raise RuntimeError(
                    "Problem adding networks. Both have specified bond details but they are not the same!!"
                )
            new_details[key] = value

        net.details_of_bondtypes = new_details

        return net


def rotate_network(network: Network, angle):
    """
    Rotates the network by a specified angle around the center of the domain.
    """
    shift_x = network.domain.sizex / 2
    shift_y = network.domain.sizey / 2

    pos = np.array(network.beads_positions)

    pos[:, 0] -= shift_x
    pos[:, 1] -= shift_y

    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    pos[:, :2] = (rotation_matrix @ (pos[:, :2].transpose())).transpose()

    pos[:, 0] += shift_x
    pos[:, 1] += shift_y

    network.beads_positions = pos.tolist()


class NetworkBuilder:
    """
    A builder class for constructing a network of beads, bonds, and angles. (Currently only used to construct the 'triangle' network.)
    """

    def __init__(self):
        self.beads_positions: List[Tuple[float, float]] = []
        self.beads_types: List[BEADTYPE] = []
        self.bonds_groups: List[BOND] = []
        self.bonds_types: List[BONDTYPE] = []
        self.angle_groups: List[ANGLE] = []
        self.angle_types: List[ANGLETYPE] = []
        self.bead_counter: int = -1

    def add_bead(self, position: Tuple[float, float], bead_type: BEADTYPE = "free"):
        """Adds a bead to the network and return the id of that bead."""
        self.beads_positions.append(position)
        self.beads_types.append(bead_type)
        self.bead_counter += 1
        return self.bead_counter

    def add_bond(self, bead1: BEADID, bead2: BEADID, bond_type: BONDTYPE = "polymer"):
        """Adds a bond between two beads."""
        self.bonds_groups.append((bead1, bead2))
        self.bonds_types.append(bond_type)

    def add_angle(
        self,
        bead1: BEADID,
        bead2: BEADID,
        bead3: BEADID,
        angle_type: ANGLETYPE = "polymer_bend",
    ):
        """Adds an angle between three beads."""
        self.angle_groups.append((bead1, bead2, bead3))
        self.angle_types.append(angle_type)

    def split_bond(
        self,
        bead1: BEADID,
        bead2: BEADID,
        create_angle: bool = False,
        angle_type: ANGLETYPE = "polymer_bend",
    ):
        """Splits a bond between two beads by creating a new bead in the middle."""
        # Calculate the midpoint of the bond
        x1, y1 = self.beads_positions[bead1]
        x2, y2 = self.beads_positions[bead2]
        new_bead_position = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Add the new bead
        new_bead_id = self.add_bead(new_bead_position)

        # Create two new bonds
        self.add_bond(bead1, new_bead_id)
        self.add_bond(new_bead_id, bead2)

        # Optionally create an angle
        if create_angle:
            self.add_angle(bead1, new_bead_id, bead2, angle_type)

    def get_network(self):
        """Returns the network's data."""
        return {
            "beads_positions": self.beads_positions,
            "beads_types": self.beads_types,
            "bonds_groups": self.bonds_groups,
            "bonds_types": self.bonds_types,
            "angle_groups": self.angle_groups,
            "angle_types": self.angle_types,
        }
