"""Moment Tensor Potential machine-learning interatomic potential."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import (
    MtpConfiguration, MtpTrainer)

LAMMPS_MTP_KOKKOS_URL = "https://github.com/RichardZJM/lammps-mtp-kokkos"


class MtpMlip(BaseMLIP):
    """Moment Tensor Potential MLIP: trained with MLIP-3 and run through the lammps-mtp-kokkos pair_style."""

    def __init__(
        self,
        mtp_trainer: MtpTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            mtp_trainer: fits the MTP and exports its LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential; must provide the mtp/extrapolation
                pair_style.
        """
        super().__init__(trainer=mtp_trainer, lammps_runner=lammps_runner)
        self._check_lammps_dependency()

    def _check_lammps_dependency(self) -> None:
        """Fail early if the LAMMPS runner lacks the mtp/extrapolation pair_style from lammps-mtp-kokkos."""
        try:
            self._lammps_runner.check_dependency(section="Pair styles", to_find="mtp/extrapolation")
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} The 'mtp/extrapolation' pair_style is provided by lammps-mtp-kokkos "
                f"({LAMMPS_MTP_KOKKOS_URL}); build LAMMPS with it using CPU-only or with Kokkos."
            ) from error

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        mtp_configuration: MtpConfiguration,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
        mlp_executable_path: Optional[Path] = None,
    ) -> "MtpMlip":
        """Reconstruct an MTP MLIP from a fitted potential file (the configuration must be provided)."""
        mtp_trainer = MtpTrainer.load_checkpoint(
            checkpoint_path, mtp_configuration=mtp_configuration, mlp_executable_path=mlp_executable_path
        )
        return cls(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        self._trainer.fit()
        self._deploy(output_directory)  # writes the potential file and caches the MtpPotential

        self._model_file = self.lammps_potential.mtp_file_path
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def write_logger_info(self, logger: logging.Logger) -> None:
        """Log the current MTP parameters."""
        logger.info("  The MTP parameters are now:")
        for name, value in self._mtp_parameters().items():
            logger.info(f"       {name} = {value}")

    def _mtp_parameters(self) -> Dict:
        """The parameters describing the current MTP model."""
        configuration = self._trainer.configuration
        return dict(
            level=configuration.level,
            max_dist=configuration.max_dist,
            min_dist=configuration.min_dist,
            radial_basis_size=configuration.radial_basis_size,
            alpha_scalar_moments=configuration.alpha_scalar_moments,
            species_count=configuration.species_count,
            number_of_adjustable_parameters=configuration.number_of_adjustable_parameters,
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        # For MTP the model, the uncertainty and the LAMMPS pair-coeff are all the same .almtp file.
        potential_file = None if potential is None else str(potential.mtp_file_path)
        return dict(
            model_file=model_file,
            unc_file=potential_file,
            lammps_potential_file=potential_file,
            hyperparameters=self._mtp_parameters(),
        )
