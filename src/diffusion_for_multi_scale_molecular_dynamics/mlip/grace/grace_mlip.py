"""GRACE-FS machine-learning interatomic potential."""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_configuration import \
    GraceConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_trainer import \
    GraceTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)

GRACE_LAMMPS_CLONE = "git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git"
GRACEMAKER_CLONE = "git clone https://github.com/ICAMS/grace-tensorpotential.git"
PACE_ACTIVESET_CLONE = "git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git"


class GraceMlip(BaseMLIP):
    """GRACE-FS MLIP: trained with gracemaker and run through the lammps grace/fs pair_style."""

    def __init__(
        self,
        grace_trainer: GraceTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            grace_trainer: fits the GRACE-FS model and exports its LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential; must provide the grace/fs pair_style.
        """
        super().__init__(trainer=grace_trainer, lammps_runner=lammps_runner)
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Fail early if the LAMMPS grace/fs pair_style or the gracemaker / pace_activeset tools are missing."""
        try:
            self._lammps_runner.check_dependency(section="Pair styles", to_find="grace/fs")
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} The grace/fs pair_style requires a GRACE-enabled LAMMPS build: {GRACE_LAMMPS_CLONE}"
            ) from error

        if shutil.which("gracemaker") is None:
            raise RuntimeError(
                f"gracemaker was not found on PATH; it is required to train a GRACE-FS model. Install it with: "
                f"{GRACEMAKER_CLONE}"
            )
        if shutil.which("pace_activeset") is None:
            raise RuntimeError(
                "pace_activeset was not found on PATH; it builds the GRACE-FS active set (.asi) used for the "
                f"extrapolation grade. Install it with: {PACE_ACTIVESET_CLONE}"
            )

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        self._trainer.fit()
        self._deploy(output_directory)  # write_checkpoint: builds the active set and caches the GracePotential

        self._model_file = self.lammps_potential.model_file_path
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log the current GRACE-FS parameters."""
        logger.info("  The GRACE-FS parameters are now:")
        for name, value in self._grace_parameters().items():
            logger.info(f"       {name} = {value}")

    def minimum_number_of_training_environments(self):
        """Minimum number of atomic environments per element needed for the D-optimality active set.

        GRACE-FS uncertainty relies on a D-optimality active set built per element: MaxVol selects the
        environments that span each element's descriptor space. The floor is therefore per element - the count
        of an element's atoms summed over the training set must be at least n_proj[element], the dimension of
        that element's descriptor space, n_proj = compute_number_of_functions(GRACEFSBasisSet(model)) (scaled
        by the per-species ndensity for the nonlinear .asi). Computing it needs the model architecture (pyace /
        GRACE python API), so it is left unimplemented for now.
        """
        pass

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        grace_configuration: GraceConfiguration,
        initial_configuration: SinglePointCalculation,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
        gracemaker_executable_path: Optional[Path] = None,
        pace_activeset_executable_path: Optional[Path] = None,
    ) -> "GraceMlip":
        """Reconstruct a GRACE-FS MLIP from a checkpoint (the configuration and initial config must be provided)."""
        grace_trainer = GraceTrainer.load_checkpoint(
            checkpoint_path,
            grace_configuration=grace_configuration,
            initial_configuration=initial_configuration,
            gracemaker_executable_path=gracemaker_executable_path,
            pace_activeset_executable_path=pace_activeset_executable_path,
        )
        return cls(grace_trainer=grace_trainer, lammps_runner=lammps_runner)

    def _grace_parameters(self) -> Dict:
        """The parameters describing the current GRACE-FS model."""
        configuration = self._trainer.configuration
        return dict(
            elements=configuration.elements,
            cutoff=configuration.cutoff,
            preset=configuration.preset,
            size=configuration.size,
            seed=configuration.seed,
            target_total_updates=configuration.target_total_updates,
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        # The FS model (.yaml) is the LAMMPS pair-coeff; the active set (.asi) provides the extrapolation grade.
        lammps_potential_file = None if potential is None else str(potential.model_file_path)
        unc_file = None if potential is None else str(potential.active_set_file_path)
        return dict(
            model_file=model_file,
            unc_file=unc_file,
            lammps_potential_file=lammps_potential_file,
            hyperparameters=self._grace_parameters(),
        )
