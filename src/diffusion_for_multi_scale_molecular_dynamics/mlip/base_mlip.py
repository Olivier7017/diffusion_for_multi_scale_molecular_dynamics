"""Base class for a trainable interatomic potential used in the active learning loop."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class BaseMLIP(ABC):
    """A trainable interatomic potential for the active learning loop."""

    @property
    @abstractmethod
    def lammps_potential(self) -> LammpsPotential:
        """The currently deployed LAMMPS potential."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Add a labelled structure to the training set."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def prepare_mlip_first_round(self, output_directory: Path) -> None:
        """Deploy the pretrained model so it can be run before any training happens this campaign."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def training_metrics(self) -> Dict:
        """Return training-set metrics: number of configurations, energy RMSE and forces RMSE."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log a summary of the current model state."""
        raise NotImplementedError("must be implemented in a child class.")

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "BaseMLIP":
        """Reconstruct the MLIP from a checkpoint."""
        raise NotImplementedError("must be implemented in a child class.")
