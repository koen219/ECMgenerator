import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class StrandDistribution(ABC):
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
    def __init__(self, sizex, sizey, seed: Optional[int] = None):
        self._sizex = sizex
        self._sizey = sizey
        if seed:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    def pos_x_dist(self, n):
        """
        Return starting positions of beads
        """
        return self._rng.uniform(low=0, high=self._sizex, size=n)

    def pos_y_dist(self, n):
        return self._rng.uniform(0, self._sizey, size=n)

    def angle_dist(self, n):
        return self._rng.uniform(0, 2 * np.pi, size=n)
