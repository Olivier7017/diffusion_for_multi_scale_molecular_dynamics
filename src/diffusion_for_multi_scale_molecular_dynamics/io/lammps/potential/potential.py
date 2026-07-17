"""Base class for interatomic potentials expressed as LAMMPS pair commands."""

from abc import ABC, abstractmethod
from typing import Optional


class LammpsPotential(ABC):
    """A fixed interatomic potential that knows how to write itself into a LAMMPS input.

    It is the simple, fixed counterpart of a trainable MLIP: it only has to emit its pair commands.
    """

    calculation_type: str = "lammps"

    @abstractmethod
    def pair_style_command(self) -> str:
        """Return the LAMMPS pair_style command."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def pair_coeff_command(self, elements_string: str) -> str:
        """Return the LAMMPS pair_coeff command."""
        raise NotImplementedError("must be implemented in a child class.")

    def uncertainty_compute_command(self) -> Optional[str]:
        """Return the per-atom uncertainty compute command, or None if the potential has no uncertainty."""
        return None
