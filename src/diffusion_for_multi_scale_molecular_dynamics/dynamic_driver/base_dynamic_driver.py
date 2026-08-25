"""Base class for the dynamic drivers (ARTn, MD) that search for uncertain structures with LAMMPS."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.dynamic_driver_lammps_input import \
    build_dynamic_driver_lammps_inputs
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    configure_logging


class DynamicDriver(ABC):
    """Drive a LAMMPS run that searches for an uncertain structure and halts when one is found.

    The shared LAMMPS script (read configuration, define the potential, watch the per-atom uncertainty and
    dump the structure when it exceeds the threshold) lives in the base; subclasses provide the dynamics
    block (ARTn minimization vs MD) and decide the resulting CalculationState.
    """

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        reference_directory: Path,
    ):
        """Init method.

        Args:
            lammps_runner: a runner that drives LAMMPS (must handle the MLIP pair_style + uncertainty).
            reference_directory: directory containing 'initial_configuration.dat' (the starting structure).
        """
        assert reference_directory.is_dir(), "The reference directory is not valid."

        self._initial_configuration_file_path = reference_directory / "initial_configuration.dat"
        assert self._initial_configuration_file_path.is_file(), "The initial configuration file does not exist."
        self.initial_structure = self._load_initial_configuration(self._initial_configuration_file_path)

        self._lammps_runner = lammps_runner
        self._lammps_input_filename = "lammps.in"

    @staticmethod
    def _load_initial_configuration(initial_configuration_file_path: Path) -> Structure:
        """Load the initial configuration as a structure."""
        try:
            structure = LammpsData.from_file(str(initial_configuration_file_path),
                                             atom_style="atomic",
                                             sort_id=True).structure
        except Exception:
            raise ValueError(f"The initial configuration file {initial_configuration_file_path} cannot be loaded.\n"
                             "Make sure the file is present and in a format that can be read by the "
                             "LammpsData module.")
        return structure

    def run(self, mlip: BaseMLIP, working_directory: Path, uncertainty_threshold: float) -> CalculationState:
        """Run the dynamics with the MLIP and return the resulting calculation state.

        Args:
            mlip: the machine-learning interatomic potential; its deployed LAMMPS potential provides the
                interaction commands and the per-atom uncertainty field.
            working_directory: directory where the run is performed (must not already exist).
            uncertainty_threshold: the run halts when the per-atom uncertainty exceeds this value.

        Returns:
            calculation_state: SUCCESS, INTERRUPTION or ERROR.
        """
        logger = self._setup_working_directory(working_directory)
        parameters = self._build_lammps_parameters(mlip, uncertainty_threshold)
        self._write_lammps_input(working_directory, parameters)
        if not self._execute_lammps(working_directory, logger):
            return CalculationState.ERROR
        return self._get_calculation_state(working_directory)

    def _setup_working_directory(self, working_directory: Path) -> logging.Logger:
        """Create the (empty) working directory, set up logging, and write the starting configuration."""
        if working_directory.is_dir() and any(working_directory.iterdir()):
            raise ValueError(
                f"The working directory {working_directory} already exists and is not empty! "
                "Exiting to avoid writing over existing data."
            )
        working_directory.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("dynamic_driver_run")
        configure_logging(experiment_dir=str(working_directory), logger=logger, log_to_console=False)

        logger.info("Write the starting configuration to the working directory.")
        lammps_data = LammpsData.from_structure(self.initial_structure, atom_style="atomic")
        lammps_data.write_file(str(working_directory / "initial_configuration.dat"))
        self._prepare_reference_files(working_directory)
        return logger

    def _build_lammps_parameters(self, mlip: BaseMLIP, uncertainty_threshold: float) -> dict:
        """Assemble the template substitution parameters from the MLIP potential and the dynamics block."""
        potential = mlip.lammps_potential
        group_block, mass_block, elements_string = generate_named_elements_blocks(self.initial_structure)
        return dict(
            configuration_file_path="initial_configuration.dat",
            interaction_commands="\n".join(potential.interaction_commands(elements_string, with_uncertainty=True)),
            uncertainty_field=potential.uncertainty_field(),
            dump_fields=" ".join(potential.dump_fields(with_uncertainty=True)),
            uncertainty_threshold=f"{uncertainty_threshold:.12f}",
            group_block=group_block,
            mass_block=mass_block,
            elements_string=elements_string,
            dynamics_block=self._dynamics_block(),
        )

    def _write_lammps_input(self, working_directory: Path, parameters: dict) -> None:
        """Build the shared LAMMPS input script from the parameters and write it to the working directory."""
        script_content = build_dynamic_driver_lammps_inputs(**parameters)
        with open(working_directory / self._lammps_input_filename, "w") as file_descriptor:
            file_descriptor.write(script_content)

    def _execute_lammps(self, working_directory: Path, logger: logging.Logger) -> bool:
        """Run LAMMPS in the working directory; return False if it failed, True otherwise."""
        logger.info("Launching LAMMPS")
        start_time = time.time()
        try:
            self._lammps_runner.run_lammps(working_directory=working_directory,
                                           lammps_input_file_name=self._lammps_input_filename)
        except RuntimeError:
            logger.exception("LAMMPS execution failed.")
            return False
        logger.info(f"LAMMPS execution has finished. Execution Time: {time.time() - start_time: 6.3e} sec.")
        return True

    @abstractmethod
    def _prepare_reference_files(self, working_directory: Path) -> None:
        """Copy any driver-specific reference files into the working directory before the run."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def _dynamics_block(self) -> str:
        """Return the LAMMPS dynamics commands rendered into the shared template's $dynamics_block."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Determine the CalculationState from the finished run's output."""
        raise NotImplementedError("must be implemented in a child class.")
