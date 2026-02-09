from __future__ import annotations

from abc import ABC, abstractmethod


class RepulsionCalculator(ABC):
    """Analytical Repulsion Calculator.

    This class calculates a score based on an analytical model to helps stability.
    This is used in LangevinGenerator predictor_step and corrector step to penalize atomic overlaps
    """

    def __init__(
        self,
        safe_radius: float
    ):
        """Init method.
        
        Args:
            safe_radius (ang): The minimal interatomic distance with no contribution from this analytical model
        """
        self.safe_radius = safe_radius

    def get_minimal_atomic_distances(self, X):
        """Return minimal interatomic distance per configuration (shape: [nconf]).

        Args:
            X : """
        raise NotImplementedError("TODO")

    @abstractmethod
    def get_analytical_score(self, X):
        """Return analytical repulsion score (shape: [nconf, natoms, 3])."""
        pass

