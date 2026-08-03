"""FLARE machine-learning interatomic potential."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
    FlareTrainer


class FlareMLIP(BaseMLIP):
    """FLARE machine-learning interatomic potential."""

    def __init__(
        self,
        flare_trainer: FlareTrainer,
        hyperparameter_optimizer: FlareHyperparametersOptimizer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            flare_trainer: wraps the FLARE sparse Gaussian process.
            hyperparameter_optimizer: drives the hyperparameter fitting during train().
            lammps_runner: runner used to evaluate the deployed potential.
        """
        super().__init__(trainer=flare_trainer, lammps_runner=lammps_runner)
        self._hyperparameter_optimizer = hyperparameter_optimizer
        self._model_file: Optional[Path] = None

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        hyperparameter_optimizer: Optional[FlareHyperparametersOptimizer] = None,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
    ) -> "FlareMLIP":
        """Reconstruct a FLARE MLIP from a checkpoint (using an inactive optimizer if none is given)."""
        flare_trainer = FlareTrainer.load_checkpoint(checkpoint_path)
        if hyperparameter_optimizer is None:
            hyperparameter_optimizer = FlareHyperparametersOptimizer(
                FlareOptimizerConfiguration(
                    optimize_sigma=False, optimize_sigma_e=False, optimize_sigma_f=False, optimize_sigma_s=False
                )
            )
        return cls(
            flare_trainer=flare_trainer,
            hyperparameter_optimizer=hyperparameter_optimizer,
            lammps_runner=lammps_runner,
        )

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        if not self._hyperparameter_optimizer.is_inactive:
            _, history_df = self._trainer.fit_hyperparameters(self._hyperparameter_optimizer)
            history_df.to_pickle(output_directory / "hyperparameter_optimization_log.pkl")

        self._trainer.fit()
        self._deploy(output_directory)

        self._model_file = output_directory / "checkpoint.json"
        self._trainer.write_checkpoint(self._model_file)
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log the current hyperparameters."""
        logger.info("  The SGP hyperparameters are now:")
        for name, value in self._flare_hyperparameters().items():
            logger.info(f"       {name} = {value: 12.8f}")

    def _flare_hyperparameters(self) -> Dict:
        """The current SGP hyperparameters (sigma, sigma_e, sigma_f, sigma_s)."""
        sigma, sigma_e, sigma_f, sigma_s = self._trainer.sgp_model.sparse_gp.hyperparameters
        return dict(
            sigma=float(sigma), sigma_e=float(sigma_e), sigma_f=float(sigma_f), sigma_s=float(sigma_s)
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        unc_file = None if potential is None else str(potential.mapped_uncertainty_file_path)
        lammps_potential_file = None if potential is None else str(potential.pair_coeff_file_path)
        hyperparameters = self._flare_hyperparameters()

        return dict(
            model_file=model_file,
            unc_file=unc_file,
            lammps_potential_file=lammps_potential_file,
            hyperparameters=hyperparameters,
        )
