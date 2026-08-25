"""Base class for a trainable interatomic potential used in the active learning loop."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_pymatgen_structure


class BaseMLIP(ABC):
    """A trainable interatomic potential for the active learning loop.

    A MLIP is built from a trainer (the trainable core) and it caches the LAMMPS potential
    that the trainer exports whenever the model is deployed.
    """

    def __init__(
        self,
        trainer: BaseMLIPTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            trainer: the trainable core that learns and exports a LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential (e.g. for training metrics).
        """
        self._trainer = trainer
        self._lammps_runner = lammps_runner
        self._lammps_potential: Optional[LammpsPotential] = None
        self._model_file: Optional[Path] = None

    @property
    def lammps_potential(self) -> LammpsPotential:
        """The currently deployed LAMMPS potential."""
        if self._lammps_potential is None:
            raise RuntimeError("The MLIP has not been deployed yet; call prepare_mlip_first_round or train first.")
        return self._lammps_potential

    @property
    def model_file(self) -> Optional[Path]:
        """Path to the current reloadable model checkpoint (None until the model has been trained)."""
        return self._model_file

    @property
    def training_database(self) -> Optional[TrainingDatabase]:
        """The training database backing this MLIP's trainer (None until one is attached)."""
        return self._trainer.training_database

    def attach_training_database(self, training_database: TrainingDatabase) -> None:
        """Attach the training database; the active learning loop calls this at the start of a run."""
        self._trainer.set_training_database(training_database)

    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Add a labelled structure to the training set."""
        self._trainer.add_labelled_structure(single_point_calculation, active_environment_indices)

    def prepare_mlip_first_round(self, output_directory: Path) -> None:
        """Deploy the pretrained model so it can be run before any training happens this campaign."""
        self._deploy(output_directory)

    def _deploy(self, output_directory: Path) -> None:
        """Write the model into output_directory and cache the resulting LAMMPS potential."""
        self._lammps_potential = self._trainer.write_checkpoint(output_directory)

    def calculate(
        self, configurations: Union[Structure, Atoms, List[Union[Structure, Atoms]]]
    ) -> List[SinglePointCalculation]:
        """Evaluate configurations with the deployed potential.

        Args:
            configurations: a single configuration (pymatgen Structure or ase.Atoms), or a list of them.

        Returns:
            a list of SinglePointCalculation.
        """
        if not isinstance(configurations, list):
            configurations = [configurations]

        calculator = LammpsSinglePointCalculator(
            lammps_potential=self.lammps_potential, lammps_runner=self._lammps_runner
        )
        return [calculator.calculate(to_pymatgen_structure(configuration)) for configuration in configurations]

    def training_metrics(self, reference_atoms: Optional[List[Atoms]] = None) -> Dict:
        """Return accuracy metrics (configuration count, energy RMSE, forces RMSE) of the deployed potential.

        The metrics are computed over the training set by default; pass reference_atoms to evaluate the
        potential against a given set of labelled ase.Atoms instead.
        """
        if reference_atoms is None:
            calculations = self._trainer.labelled_calculations
        else:
            calculations = [
                SinglePointCalculation(calculation_type="reference",
                                       structure=to_pymatgen_structure(atoms),
                                       forces=atoms.get_forces(),
                                       energy=atoms.get_potential_energy())
                for atoms in reference_atoms
            ]

        if not calculations:  # e.g. round 0 of active learning
            return dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)

        predictions = self.calculate([calculation.structure for calculation in calculations])

        energy_errors = []
        force_errors = []
        for prediction, calculation in zip(predictions, calculations):
            energy_errors.append(prediction.energy - calculation.energy)
            force_errors.append((np.asarray(prediction.forces) - np.asarray(calculation.forces)).ravel())

        rmse_energy = float(np.sqrt(np.mean(np.square(energy_errors))))
        rmse_forces = float(np.sqrt(np.mean(np.square(np.concatenate(force_errors)))))
        return dict(n_training_conf=len(calculations), rmse_energy=rmse_energy, rmse_forces=rmse_forces)

    @abstractmethod
    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
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
