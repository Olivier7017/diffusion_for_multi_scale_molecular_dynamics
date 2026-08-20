"""MD dynamic driver."""

from pathlib import Path
from string import Template
from typing import Union

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.io.artn import \
    CalculationState

PATH_TO_MD_TEMPLATE = Path(__file__).parent / "md.template"
_VELOCITY_SEED = 12345


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

        with open(PATH_TO_MD_TEMPLATE, mode="r") as file_descriptor:
            self._dynamics_template = Template(file_descriptor.read())

    def _prepare_reference_files(self, working_directory: Path) -> None:
        """MD needs no reference file beyond the starting configuration."""

    def _dynamics_block(self) -> str:
        """Render the NVT MD dynamics commands (velocity + Nose-Hoover thermostat + run)."""
        return self._dynamics_template.safe_substitute(
            temperature=self._temperature,
            velocity_seed=_VELOCITY_SEED,
            thermostat_damping=100.0 * self._timestep,
            timestep=self._timestep,
            number_of_steps=self._number_of_steps,
        )

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Interpret the run: a non-empty uncertain dump means the halt fired on an uncertain structure."""
        uncertain_dump_path = working_directory / "uncertain_dump.yaml"
        if uncertain_dump_path.is_file() and uncertain_dump_path.stat().st_size > 0:
            return CalculationState.INTERRUPTION
        return CalculationState.SUCCESS
