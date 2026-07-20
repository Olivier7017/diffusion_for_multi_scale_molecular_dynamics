"""FLARE machine-learning interatomic potential."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.calc.flare_single_point_calculator import \
    FlareSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.flare import \
    FlarePotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
    FlareTrainer


class FlareMLIP(BaseMLIP):
    """FLARE machine-learning interatomic potential."""

    def __init__(self, flare_trainer: FlareTrainer, hyperparameter_optimizer: FlareHyperparametersOptimizer):
        """Init method.

        Args:
            flare_trainer: wraps the FLARE sparse Gaussian process.
            hyperparameter_optimizer: drives the hyperparameter fitting during train().
        """
        self._flare_trainer = flare_trainer
        self._hyperparameter_optimizer = hyperparameter_optimizer

        self._labelled_calculations: List[SinglePointCalculation] = []
        self._lammps_potential: Optional[FlarePotential] = None
        self._model_file: Optional[Path] = None
        self._lammps_potential_file: Optional[Path] = None
        self._uncertainty_file: Optional[Path] = None
        self._deploy_version = 0

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: Path, hyperparameter_optimizer: Optional[FlareHyperparametersOptimizer] = None
    ) -> "FlareMLIP":
        """Reconstruct a FLARE MLIP from a checkpoint (using an inactive optimizer if none is given)."""
        flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)
        if hyperparameter_optimizer is None:
            hyperparameter_optimizer = FlareHyperparametersOptimizer(
                FlareOptimizerConfiguration(
                    optimize_sigma=False, optimize_sigma_e=False, optimize_sigma_f=False, optimize_sigma_s=False
                )
            )
        return cls(flare_trainer=flare_trainer, hyperparameter_optimizer=hyperparameter_optimizer)

    @property
    def lammps_potential(self) -> FlarePotential:
        """The currently deployed LAMMPS potential."""
        if self._lammps_potential is None:
            raise RuntimeError("The MLIP has not been deployed yet; call prepare_mlip_first_round or train first.")
        return self._lammps_potential

    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Add a labelled structure to the training set."""
        self._flare_trainer.add_labelled_structure(single_point_calculation, active_environment_indices)
        self._labelled_calculations.append(single_point_calculation)

    def prepare_mlip_first_round(self, output_directory: Path) -> None:
        """Deploy the pretrained model so it can be run before any training happens this campaign."""
        self._deploy(output_directory)

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        if not self._hyperparameter_optimizer.is_inactive:
            _, history_df = self._flare_trainer.fit_hyperparameters(self._hyperparameter_optimizer)
            history_df.to_pickle(output_directory / "hyperparameter_optimization_log.pkl")

        self._flare_trainer.fit()
        self._deploy(output_directory)

        self._model_file = output_directory / "checkpoint.json"
        self._flare_trainer.write_checkpoint_to_disk(self._model_file)
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log the current hyperparameters."""
        logger.info("  The SGP hyperparameters are now:")
        for name, value in self._state()["hyperparameters"].items():
            logger.info(f"       {name} = {value: 12.8f}")

    def training_metrics(self) -> Dict:
        """Return training-set metrics: number of configurations, energy RMSE and forces RMSE."""
        number_of_configurations = len(self._labelled_calculations)
        if number_of_configurations == 0:
            return dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)

        calculator = FlareSinglePointCalculator(self._flare_trainer.sgp_model)
        energy_errors = []
        force_errors = []
        for calculation in self._labelled_calculations:
            prediction = calculator.calculate(calculation.structure)
            energy_errors.append(prediction.energy - calculation.energy)
            force_errors.append((np.asarray(prediction.forces) - np.asarray(calculation.forces)).ravel())

        rmse_energy = float(np.sqrt(np.mean(np.square(energy_errors))))
        rmse_forces = float(np.sqrt(np.mean(np.square(np.concatenate(force_errors)))))
        return dict(n_training_conf=number_of_configurations, rmse_energy=rmse_energy, rmse_forces=rmse_forces)

    def _deploy(self, output_directory: Path) -> None:
        """Write the mapped LAMMPS files and build the corresponding potential."""
        self._deploy_version += 1
        pair_coeff_file_path, uncertainty_file_path = self._flare_trainer.write_mapped_model_to_disk(
            output_directory, version=self._deploy_version
        )
        self._lammps_potential_file = pair_coeff_file_path
        self._uncertainty_file = uncertainty_file_path
        self._lammps_potential = FlarePotential(
            pair_coeff_file_path=pair_coeff_file_path, mapped_uncertainty_file_path=uncertainty_file_path
        )

    def _state(self) -> Dict:
        model_file = None if self._model_file is None else str(self._model_file)
        unc_file = None if self._uncertainty_file is None else str(self._uncertainty_file)
        lammps_potential_file = None if self._lammps_potential_file is None else str(self._lammps_potential_file)

        sigma, sigma_e, sigma_f, sigma_s = self._flare_trainer.sgp_model.sparse_gp.hyperparameters
        hyperparameters = dict(
            sigma=float(sigma), sigma_e=float(sigma_e), sigma_f=float(sigma_f), sigma_s=float(sigma_s)
        )

        return dict(
            model_file=model_file,
            unc_file=unc_file,
            lammps_potential_file=lammps_potential_file,
            hyperparameters=hyperparameters,
        )
