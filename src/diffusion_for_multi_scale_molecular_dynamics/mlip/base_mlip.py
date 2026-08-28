"""Base class for a trainable interatomic potential used in the active learning loop."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    create_perturbed_structures


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
        # Specific descriptor information; populated by each MLIP subclass.
        self.descriptors: Dict[str, Any] = {}

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

    def minimum_number_of_training_environments(self) -> int:
        """Number of atomic environments the active set needs (the descriptor-space dimension).

        The default is 0 (no D-optimality active set, e.g. FLARE); a D-optimality MLIP (MTP, GRACE-FS)
        redefines this.
        """
        return 0

    def minimum_number_of_atomic_structures(self, structure: Atoms, number_of_existing_environments: int = 0) -> int:
        """Number of structures of this composition needed to reach the minimum number of training environments."""
        missing_environments = self.minimum_number_of_training_environments() - number_of_existing_environments
        if missing_environments <= 0:
            return 0
        return int(np.ceil(missing_environments / len(structure)))

    def augment_configurations(
        self, structure: Atoms, number_of_existing_environments: int = 0, standard_deviation: float = 0.05
    ) -> List[Atoms]:
        """Perturb the structure into enough copies to top the training set up to the D-optimality minimum.

        Args:
            structure: the seed configuration to perturb.
            number_of_existing_environments: environments already in the training set.
            standard_deviation: standard deviation (Angstrom) of the Gaussian displacements.

        Returns:
            the perturbed configurations sized to reach the minimum (empty if already met).
        """
        return create_perturbed_structures(
            structure,
            standard_deviation,
            self.minimum_number_of_atomic_structures(structure, number_of_existing_environments),
        )

    def prepare_mlip_first_round(self, output_directory: Path) -> None:
        """Deploy the pretrained model so it can be run before any training happens this campaign."""
        self._deploy(output_directory)

    def _deploy(self, output_directory: Path) -> None:
        """Write the model into output_directory and cache the resulting LAMMPS potential."""
        self._lammps_potential = self._trainer.write_checkpoint(output_directory)

    def load(self, model_directory: Path) -> None:
        """Load an already-trained model from model_directory into a runnable potential (no fitting).

        Used to resume a campaign straight off disk. Retraining later refits from the training database, so
        this only needs to restore the deployed potential, not the trainer's state.
        """
        raise NotImplementedError("must be implemented in a child class.")

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
