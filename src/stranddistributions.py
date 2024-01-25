import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod


class StrandDistribution(ABC):
    def __init__(self, rng):
        self._rng = rng

    @abstractmethod
    def pos_x_dist(self, n) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def pos_y_dist(self, n) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def angle_dist(self, n) -> npt.NDArray[np.float64]:
        pass


class UniformStrandDistribution(StrandDistribution):
    def __init__(self, rng, Lx, Ly):
        super().__init__(rng)
        self._Lx = Lx
        self._Ly = Ly

    def pos_x_dist(self, n):
        """
        Return starting positions of beads
        """
        return self._rng.uniform(0, 2*self._Lx, size=n)

    def pos_y_dist(self, n):
        return self._rng.uniform(0, 2*self._Ly, size=n)

    def angle_dist(self, n):
        return self._rng.uniform(0, 2 * np.pi, size=n)
