from .network import Network, BEADTYPE, BONDTYPE
from .stranddistributions import StrandDistribution
from .parameters import Parameter

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional

class StrandGenerator(ABC):
    def __init__(self, network: Optional[Network] = None):
        self._network: Network

        if network:
            self._network = network
        else:
            self._network = Network()

    @abstractmethod
    def build_strands(self, par) -> Network:
        pass



class RandomStrandGenerator(StrandGenerator):
    def __init__(
        self, strand_distribution: StrandDistribution, network: Optional[Network] = None
    ):
        self._strand_distribution = strand_distribution
        super().__init__(network)


    def build_strands(self, par: Parameter):
        particlepos, types = self._pos_gen(par)
        bondsgroup, bondstypes = self._bond_gen(par)
        anglegroup, angletypes = self._angle_gen(par)
        
        self._network.beads_positions.extend(particlepos) 
        self._network.beads_types.extend(types) 
        
        self._network.bonds_groups.extend(bondsgroup)
        self._network.bonds_types.extend(bondstypes)
        
        self._network.angle_groups.extend(anglegroup)
        self._network.angle_types.extend(angletypes)
        
        return self._network

    def _pos_gen(self, par: Parameter):
        num_particles = par.number_of_strands * par.number_of_beads_per_strand
        num_strands = par.number_of_strands
        num_beads = par.number_of_beads_per_strand
        contour_length = par.contour_length_of_strand

        dist = self._strand_distribution

        middle_of_strand = num_beads // 2

        pos = np.zeros((num_strands, num_beads, 2))
        pos[:, middle_of_strand, 0] = dist.pos_x_dist(num_strands)
        pos[:, middle_of_strand, 1] = dist.pos_y_dist(num_strands)

        h = contour_length / (num_beads - 1)

        angles = dist.angle_dist(num_strands)
        for bead in range(num_beads):
            if bead == middle_of_strand:
                continue
            coss = np.cos(angles)
            sins = np.sin(angles)
            v = h * np.column_stack([coss, sins])
            pos[:, bead, :] = pos[:, middle_of_strand, :] + v * (middle_of_strand - bead)
        pos = pos.reshape((num_particles, 2))
        typeid = np.array(["free"] * num_particles, dtype=object)

        if par.fix_boundary:
            boundary_particles = (abs(pos[:, 0]) > par.Lx) | (abs(pos[:, 1]) > par.Ly)
            typeid[boundary_particles] = "boundary"

        typeid = typeid.tolist()

        return pos, typeid

    def _bond_gen(self, par):
        num_strands = par.number_of_strands
        num_beads = par.number_of_beads_per_strand
        bondsgroup = np.empty(shape=(num_strands * (num_beads - 1), 2), dtype=int)
        bonds_indices = np.repeat(
            num_beads * np.arange(0, num_strands, dtype=int), num_beads - 1
        )
        bondsgroup[:, 0] = bonds_indices + np.tile(
            np.arange(0, num_beads - 1, 1, dtype=int), num_strands
        )
        bondsgroup[:, 1] = bonds_indices + np.tile(
            np.arange(1, num_beads, 1, dtype=int), num_strands
        )
        bondsgroup = bondsgroup.tolist()
        bondstype = ["polymer"] * len(bondsgroup)

        return bondsgroup, bondstype

    def _angle_gen(self, par):
        num_strands = par.number_of_strands
        num_beads = par.number_of_beads_per_strand
        angles_group = np.empty(shape=(num_strands * (num_beads - 2), 3), dtype=int)
        angles_indices = np.repeat(
            num_beads * np.arange(0, num_strands, dtype=int), num_beads - 2
        )
        angles_group[:, 0] = angles_indices + np.tile(
            np.arange(0, num_beads - 2, 1, dtype=int), num_strands
        )
        angles_group[:, 1] = angles_indices + np.tile(
            np.arange(1, num_beads - 1, 1, dtype=int), num_strands
        )
        angles_group[:, 2] = angles_indices + np.tile(
            np.arange(2, num_beads, 1, dtype=int), num_strands
        )
        anglegroup = angles_group.tolist()
        types = ["polymer_bend"] * len(anglegroup)
        return anglegroup, types
