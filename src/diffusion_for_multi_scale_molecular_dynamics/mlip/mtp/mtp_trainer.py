"""Moment Tensor Potential trainer."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pymatgen.core import Element

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.mtp import \
    MtpPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer


@dataclass(kw_only=True)
class MtpConfiguration:
    """A Moment Tensor Potential model, trained with MLIP-3 and run through the lammps-mtp-kokkos interface."""

    elements: list[str]

    # Inputs that define the model.
    level: int
    max_dist: float

    # Read back from the template/fitted potential (the level fixes these).
    min_dist: float = 0.0
    radial_basis_size: Optional[int] = None
    alpha_scalar_moments: Optional[int] = None
    species_count: Optional[int] = None

    energy_weight: float = 0.0
    force_weight: float = 0.0
    stress_weight: float = 0.0
    site_en_weight: float = 1.0

    # Parameters passed to the MLIP-3 'mlp train' command.
    training_params: dict = field(
        default_factory=lambda: dict(
            max_iter=1000, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3
        )
    )

    # The parameters read from an MTP file and their python type.
    _FILE_PARAMETERS = dict(
        species_count=int,
        min_dist=float,
        radial_basis_size=int,
        alpha_scalar_moments=int,
    )

    @property
    def number_of_adjustable_parameters(self) -> int:
        """The number of adjustable MTP parameters."""
        return self.radial_basis_size + self.alpha_scalar_moments + self.species_count

    def read_from_file(self, mtp_file_path: Path) -> None:
        """Read the level-determined parameters back from an MTP file into the configuration.

        The MTP file mixes a readable text header with binary data, so it is parsed line by line.
        """
        found = {}
        with open(mtp_file_path, "rb") as file_descriptor:
            for raw_line in file_descriptor:
                key, separator, value = raw_line.decode("latin-1").partition("=")
                key = key.strip()
                if separator and key in self._FILE_PARAMETERS and key not in found:
                    found[key] = self._FILE_PARAMETERS[key](value.strip())
                    if len(found) == len(self._FILE_PARAMETERS):
                        break

        self.species_count = found["species_count"]
        self.min_dist = found["min_dist"]
        self.radial_basis_size = found["radial_basis_size"]
        self.alpha_scalar_moments = found["alpha_scalar_moments"]

    def write_to_file(self, mtp_file_path: Path) -> None:
        """Write the configuration's max_dist into an MTP file, leaving the rest (including binary) untouched."""
        with open(mtp_file_path, "rb") as file_descriptor:
            lines = file_descriptor.readlines()

        for index, raw_line in enumerate(lines):
            decoded_line = raw_line.decode("latin-1")
            key, separator, _ = decoded_line.partition("=")
            if separator and key.strip() == "max_dist":
                leading_whitespace = decoded_line[
                    : len(decoded_line) - len(decoded_line.lstrip())
                ]
                lines[index] = (
                    f"{leading_whitespace}max_dist = {self.max_dist}\n".encode(
                        "latin-1"
                    )
                )
                break

        with open(mtp_file_path, "wb") as file_descriptor:
            file_descriptor.writelines(lines)

    def __post_init__(self):
        """Validate the configuration."""
        if len(self.elements) == 0:
            raise ValueError("The list of elements should not be empty.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("The elements are not unique!")
        for element in self.elements:
            try:
                Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")

        if self.level <= 0:
            raise ValueError("The MTP level should be positive.")
        if self.max_dist <= 0.0:
            raise ValueError("The maximum distance (cutoff) should be positive.")
        if self.min_dist < 0.0:
            raise ValueError("The minimum distance should be non-negative.")

        weights = dict(
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            stress_weight=self.stress_weight,
            site_en_weight=self.site_en_weight,
        )
        for weight_name, weight_value in weights.items():
            if weight_value < 0.0:
                raise ValueError(f"The {weight_name} should be non-negative.")


class MtpTrainer(BaseMLIPTrainer):
    """Fit a Moment Tensor Potential with the MLIP-3 'mlp train' command and deploy it as a LAMMPS potential."""

    def __init__(self, mtp_configuration: MtpConfiguration, mlp_executable_path: Optional[Path] = None):
        """Init method.

        Args:
            mtp_configuration: the MTP model definition (level, cutoff, weights, training parameters).
            mlp_executable_path: path to the MLIP-3 'mlp' executable; if None, it is looked up on PATH.
        """
        super().__init__()
        if mlp_executable_path is None:
            mlp_executable_path = shutil.which("mlp")
        if mlp_executable_path is None:
            raise ValueError("No mlp executable was provided and none was found on PATH ('which mlp' failed).")
        mlp_executable_path = Path(mlp_executable_path)
        if not mlp_executable_path.is_file():
            raise ValueError(f"The mlp executable '{mlp_executable_path}' does not exist. Review input.")

        self._configuration = mtp_configuration
        self._mlp_executable_path = mlp_executable_path
        self._fitted_potential: Optional[bytes] = None

    @property
    def configuration(self) -> MtpConfiguration:
        """The MTP configuration, with the level-determined parameters filled in once fitted."""
        return self._configuration

    def _add_labelled_structure_to_model(
        self,
        single_point_calculation: SinglePointCalculation,
        active_environment_indices: List[int],
    ) -> None:
        """Nothing is folded in per structure: MTP is fit in batch, over all labelled structures, at fit time."""
        if single_point_calculation.uncertainties is not None:
            raise ValueError(
                "Only ground-truth single-point calculations (without uncertainties) can be added."
            )

    def fit(self) -> None:
        """Fit the MTP with MLIP-3 and read the level-determined parameters back into the configuration."""
        from maml.apps.pes import MTPotential
        from maml.utils import check_structures_forces_stresses, pool_from

        if not self._labelled_calculations:
            raise RuntimeError("Cannot fit an MTP with no labelled structures.")

        structures = [
            calculation.structure for calculation in self._labelled_calculations
        ]
        forces = [
            np.asarray(calculation.forces)
            for calculation in self._labelled_calculations
        ]
        energies = [calculation.energy for calculation in self._labelled_calculations]
        checked_structures, checked_forces, _ = check_structures_forces_stresses(
            structures, forces, None
        )
        training_pool = pool_from(checked_structures, energies, checked_forces)

        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as work_directory:
            os.chdir(work_directory)
            try:
                configuration_writer = MTPotential()
                configuration_writer.elements = self._configuration.elements
                configuration_writer.write_cfg("train.cfg", cfg_pool=training_pool)

                template = Path(f"{self._configuration.level:02d}.almtp")
                shutil.copy(self._template_path(), template)
                self._configuration.write_to_file(template)

                fitted_potential_path = Path("pot.almtp")
                self._run_mlp_train(template, Path("train.cfg"), fitted_potential_path)

                self._fitted_potential = fitted_potential_path.read_bytes()
                self._configuration.read_from_file(fitted_potential_path)
            finally:
                os.chdir(original_directory)

    def write_checkpoint(self, output_directory: Path) -> MtpPotential:
        """Write the fitted potential into output_directory and return the deployed LAMMPS potential."""
        if self._fitted_potential is None:
            raise RuntimeError("The MTP has not been fitted yet; call fit first.")

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        mtp_file_path = output_directory / "potential.almtp"
        mtp_file_path.write_bytes(self._fitted_potential)
        return MtpPotential(mtp_file_path=mtp_file_path)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        mtp_configuration: MtpConfiguration,
        mlp_executable_path: Optional[Path] = None,
    ) -> "MtpTrainer":
        """Rebuild a trainer from a fitted MTP file.

        The MTP file does not record the level or the element symbols, so the configuration must be provided.
        """
        trainer = cls(
            mtp_configuration=mtp_configuration, mlp_executable_path=mlp_executable_path
        )
        trainer._fitted_potential = Path(checkpoint_path).read_bytes()
        trainer._configuration.read_from_file(Path(checkpoint_path))
        return trainer

    def _template_path(self) -> Path:
        """Path to the level template shipped with the package."""
        return (
            Path(__file__).parent.parent
            / "MTP_templates"
            / f"{self._configuration.level:02d}.almtp"
        )

    def _run_mlp_train(
        self,
        template_path: Path,
        training_configuration_path: Path,
        fitted_potential_path: Path,
    ) -> None:
        """Run 'mlp train' in the current working directory, surfacing its output on failure."""
        training_params = self._configuration.training_params
        commands = [
            str(self._mlp_executable_path),
            "train",
            str(template_path),
            str(training_configuration_path),
            f"--save_to={fitted_potential_path}",
            f"--iteration_limit={training_params['max_iter']}",
            "--al_mode=nbh",
            f"--energy-weight={self._configuration.energy_weight}",
            f"--force-weight={self._configuration.force_weight}",
            f"--stress-weight={self._configuration.stress_weight}",
            f"--site-en-weight={self._configuration.site_en_weight}",
            f"--init-params={training_params['init_params']}",
            f"--scale-by-force={training_params['scale_by_force']}",
            f"--bfgs-conv-tol={training_params['bfgs_conv_tol']}",
        ]
        result = subprocess.run(
            commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mlp train failed (exit code {result.returncode}). Command: {' '.join(commands)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
