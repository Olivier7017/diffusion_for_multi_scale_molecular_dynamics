"""ARTn dynamic driver."""

import os
import shutil
from pathlib import Path
from string import Template
from typing import Optional, Union

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.io.artn import (
    CalculationState, get_calculation_state_from_artn_output)

PATH_TO_ARTN_TEMPLATE = Path(__file__).parent / "artn.template"
ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE = "ARTN_PLUGIN_PATH"
ARTN_LIBRARY_FILE_NAME = "libartn-lmp.so"


class ArtnDriver(DynamicDriver):
    """Drive an ARTn saddle search with LAMMPS, halting when an uncertain structure is found."""

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        reference_directory: Path,
        artn_library_plugin_path: Optional[Path] = None,
    ):
        """Init method.

        Args:
            lammps_runner: a runner whose LAMMPS executable can handle ARTn and the MLIP pair_style.
            reference_directory: directory with 'initial_configuration.dat' and 'artn.in'.
            artn_library_plugin_path: path to the compiled ARTn library plugin. When None, it is read from
                the ARTN_PLUGIN_PATH environment variable. A directory is accepted too, in which case
                'libartn-lmp.so' or 'lib/libartn-lmp.so' is looked up inside it.
        """
        super().__init__(lammps_runner, reference_directory)

        self._artn_library_plugin_path = self._resolve_artn_library_plugin_path(artn_library_plugin_path)

        self._reference_artn_in_file_path = reference_directory / "artn.in"
        assert self._reference_artn_in_file_path.is_file(), "The reference artn.in file does not exist."

        with open(PATH_TO_ARTN_TEMPLATE, mode="r") as file_descriptor:
            self._dynamics_template = Template(file_descriptor.read())

    @staticmethod
    def _resolve_artn_library_plugin_path(artn_library_plugin_path: Optional[Path]) -> Path:
        """Return the plugin path, falling back to the ARTN_PLUGIN_PATH environment variable."""
        if artn_library_plugin_path is None:
            environment_value = os.environ.get(ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE)
            if environment_value is None:
                raise ValueError(
                    f"The {ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE} environment variable is not defined and "
                    "artn_library_plugin_path was not passed to ArtnDriver. Provide one of the two so the "
                    "compiled ARTn plugin can be located. The ARTn plugin can be obtained from "
                    "https://gitlab.com/mammasmias/artn-plugin."
                )
            artn_library_plugin_path = Path(environment_value)

        if artn_library_plugin_path.is_dir():
            candidate_paths = [
                artn_library_plugin_path / ARTN_LIBRARY_FILE_NAME,
                artn_library_plugin_path / "lib" / ARTN_LIBRARY_FILE_NAME,
            ]
            artn_library_plugin_path = next(
                (candidate for candidate in candidate_paths if candidate.is_file()), artn_library_plugin_path
            )

        assert artn_library_plugin_path.is_file(), "The artn library plugin_path is not valid."
        return artn_library_plugin_path

    def _prepare_reference_files(self, working_directory: Path) -> None:
        """Copy the reference artn.in into the working directory."""
        shutil.copy(self._reference_artn_in_file_path, str(working_directory / "artn.in"))

    def _dynamics_block(self) -> str:
        """Render the ARTn dynamics commands (plugin load + fix artn + FIRE minimization)."""
        return self._dynamics_template.safe_substitute(
            artn_library_plugin_path=str(self._artn_library_plugin_path)
        )

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Parse artn.out for the ARTn outcome (ERROR if the file is missing)."""
        artn_output_file_path = working_directory / "artn.out"
        if not artn_output_file_path.is_file():
            return CalculationState.ERROR
        with open(artn_output_file_path, "r") as file_descriptor:
            return get_calculation_state_from_artn_output(file_descriptor.read())
