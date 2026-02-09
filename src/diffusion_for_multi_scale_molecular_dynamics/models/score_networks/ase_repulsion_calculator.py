from __future__ import annotations
from abc import ABC, abstractmethod

from ase.calculators.calculator import Calculator

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsion_calculator import \
    RepulsionCalculator

class AseRepulsionCalculator(RepulsionCalculator):
    def __init__(
        self,
        safe_radius: float,
        calculator: Calculator
    ):
        super().__init__(safe_radius=safe_radius)
        self.calculator = calculator

    def get_analytical_score(self, X):
        raise NotImplementedError("TODO")
