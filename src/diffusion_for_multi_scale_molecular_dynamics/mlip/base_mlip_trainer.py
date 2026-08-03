"""Base class for the trainable core of an MLIP."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class BaseMLIPTrainer(ABC):
    """The trainable core of an MLIP: it learns from labelled structures and exports a LAMMPS potential."""

    def __init__(self):
        """Init method."""
        self._labelled_calculations: List[SinglePointCalculation] = []

    @property
    def labelled_calculations(self) -> List[SinglePointCalculation]:
        """The labelled calculations added to the training set through this trainer."""
        return self._labelled_calculations

    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Add a labelled structure to the training set."""
        self._add_labelled_structure_to_model(single_point_calculation, active_environment_indices)
        self._labelled_calculations.append(single_point_calculation)

    @abstractmethod
    def _add_labelled_structure_to_model(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Fold a labelled structure into the underlying model."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def fit(self) -> None:
        """Fit the model on the current training set."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_lammps_potential(self, output_directory: Path) -> LammpsPotential:
        """Write the deployable LAMMPS files into output_directory and return the matching potential."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_checkpoint(self, checkpoint_path: Path) -> None:
        """Write a checkpoint from which the trainer can be reconstructed."""
        raise NotImplementedError("must be implemented in a child class.")

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "BaseMLIPTrainer":
        """Reconstruct the trainer from a checkpoint."""
        raise NotImplementedError("must be implemented in a child class.")
