"""MD dynamic driver."""

from pathlib import Path
from typing import Union

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.md import \
    build_md_lammps_tail
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)


class MdDriver(DynamicDriver):
    """Drive an NVT molecular-dynamics run with LAMMPS, halting when an uncertain structure is found."""

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        reference_directory: Path,
        temperature: float,
        timestep: float,
        number_of_steps: int,
    ):
        """Init method.

        Args:
            lammps_runner: a runner whose LAMMPS executable can handle the MLIP pair_style.
            reference_directory: directory with 'initial_configuration.dat'.
            temperature: NVT thermostat temperature (K).
            timestep: MD timestep (ps, LAMMPS 'metal' units).
            number_of_steps: number of MD steps to run (if the uncertainty stays below the threshold).
        """
        super().__init__(lammps_runner, reference_directory)
        self._temperature = temperature
        self._timestep = timestep
        self._number_of_steps = number_of_steps

    def _prepare_reference_files(self, working_directory: Path) -> None:
        """MD needs no reference file beyond the starting configuration."""

    def _dynamics_block(self) -> str:
        """Build the NVT MD dynamics commands (velocity + Nose-Hoover thermostat + run)."""
        return build_md_lammps_tail(self._temperature, self._timestep, self._number_of_steps)

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Interpret the run: a non-empty uncertain dump means the halt fired on an uncertain structure."""
        uncertain_dump_path = working_directory / "uncertain_dump.yaml"
        if uncertain_dump_path.is_file() and uncertain_dump_path.stat().st_size > 0:
            return CalculationState.INTERRUPTION
        return CalculationState.SUCCESS
