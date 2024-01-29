from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Set
from .network import Network
from .network import BOND, BONDTYPE, BEADID, FIBREID, BONDID, Network


class CrosslinkDistributer(ABC):
    def distribute_crosslinkers(self, network: Network):
        bonds_to_add = list()
        types_to_add = list()

        beads_with_crosslinker = set()

        selected_bonds_and_types = self.select_bonds(network)

        for bond, bond_typ in selected_bonds_and_types:
            sorted_bond: List[int] = sorted(bond)

            if (
                sorted_bond in bonds_to_add
                or sorted_bond[0] in beads_with_crosslinker
                or sorted_bond[1] in beads_with_crosslinker
            ):
                continue

            bonds_to_add.append(bond)
            types_to_add.append(bond_typ)
            beads_with_crosslinker.add(bond[0])
            beads_with_crosslinker.add(bond[1])

        network.bonds_groups.extend(bonds_to_add)
        network.bonds_types.extend(types_to_add)

    @abstractmethod
    def select_bonds(self, network: Network) -> List[Tuple[BOND, BONDTYPE]]:
        pass
