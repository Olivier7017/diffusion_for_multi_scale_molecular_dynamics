"""GRACE-FS machine-learning interatomic potential."""

import logging
import shutil
from pathlib import Path
from typing import Union

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer

GRACE_LAMMPS_CLONE = "git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git"
GRACEMAKER_CLONE = "git clone https://github.com/ICAMS/grace-tensorpotential.git"
PACE_ACTIVESET_CLONE = "git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git"


class GraceMlip(BaseMLIP):
    """GRACE-FS MLIP: trained with gracemaker and run through the lammps grace/fs pair_style."""

    def __init__(
        self,
        grace_trainer: BaseMLIPTrainer,
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
        raise NotImplementedError("must be implemented in a future commit.")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        raise NotImplementedError("must be implemented in a future commit.")

    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log a summary of the current model state."""
        raise NotImplementedError("must be implemented in a future commit.")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "GraceMlip":
        """Reconstruct the MLIP from a checkpoint."""
        raise NotImplementedError("must be implemented in a future commit.")
