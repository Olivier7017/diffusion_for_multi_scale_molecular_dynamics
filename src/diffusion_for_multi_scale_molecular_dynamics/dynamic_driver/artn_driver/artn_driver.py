"""ARTn dynamic driver."""

import os
from pathlib import Path
from typing import List, Optional, Union

from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import (
    MACRO_STAGE_DESCRIPTIONS, MICRO_STAGE_DESCRIPTIONS, build_artn_lammps_tail,
    collect_artn_run_information, get_calculation_state_from_artn_output,
    write_artn_input_file)
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)

ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE = "ARTN_PLUGIN_PATH"
ARTN_LIBRARY_FILE_NAME = "libartn-lmp.so"


class ArtnDriver(DynamicDriver):
    """Drive an ARTn saddle search with LAMMPS, halting when an uncertain structure is found."""

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        initial_configuration: Atoms,
        push_ids: int,
        push_add_const: List[float],
        artn_library_plugin_path: Optional[Path] = None,
        **artn_parameters,
    ):
        """Init method.

        Args:
            lammps_runner: a runner whose LAMMPS executable can handle ARTn and the MLIP pair_style.
            initial_configuration: the single starting configuration the ARTn search is launched from.
            push_ids: the atom index ARTn pushes to escape the initial basin.
            push_add_const: the four-component push constraint for that atom.
            artn_library_plugin_path: path to the compiled ARTn library plugin. When None, it is read from
                the ARTN_PLUGIN_PATH environment variable. A directory is accepted too, in which case
                'libartn-lmp.so' or 'lib/libartn-lmp.so' is looked up inside it.
            artn_parameters: any other ARTn namelist overrides, forwarded to write_artn_input_file.
        """
        super().__init__(lammps_runner, initial_configuration)

        self._artn_library_plugin_path = self._resolve_artn_library_plugin_path(artn_library_plugin_path)
        self._push_ids = push_ids
        self._push_add_const = push_add_const
        self._artn_parameters = artn_parameters

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

    def summarize_run(self, working_directory: Path, step: int) -> List[str]:
        """Build the ARTn run-summary log lines from artn.out (step counts, interruption stage, eigenvalue).

        step is the interrupted LAMMPS step the campaign extracted; it is the true energy-evaluation count
        (artn.out's own evalf only updates once per ARTn step, so it would undercount).
        """
        information = collect_artn_run_information(working_directory)
        macro_stage, micro_stage = information["macro_stage"], information["micro_stage"]
        macro_description = MACRO_STAGE_DESCRIPTIONS.get(macro_stage, macro_stage)
        micro_description = MICRO_STAGE_DESCRIPTIONS.get(micro_stage, micro_stage)

        eigenvalue = information["lowest_eigenvalue"]
        eigenvalue_text = "n/a" if eigenvalue is None else f"{eigenvalue:.4f} eV/A^2"
        return [
            f"{information['artn_step']} ARTn step, {step} force evaluation.",
            f"Interrupted in {macro_stage} {micro_stage} ({macro_description}, {micro_description}).",
            f"Found {information['number_of_transition_pathways']} transition pathway before interruption.",
            f"lowest_eigval {eigenvalue_text}, eigenvector stability a1 {information['eigenvector_stability']:.2f}.",
        ]

    def _prepare_reference_files(self, working_directory: Path) -> None:
        """Write the ARTn input file (artn.in) into the working directory."""
        write_artn_input_file(
            working_directory / "artn.in",
            push_ids=self._push_ids,
            push_add_const=self._push_add_const,
            artn_parameters=self._artn_parameters,
        )

    def _dynamics_block(self) -> str:
        """Build the ARTn dynamics commands (plugin load + fix artn + FIRE minimization)."""
        return build_artn_lammps_tail(self._artn_library_plugin_path)

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Parse artn.out for the ARTn outcome (ERROR if the file is missing)."""
        artn_output_file_path = working_directory / "artn.out"
        if not artn_output_file_path.is_file():
            return CalculationState.ERROR
        with open(artn_output_file_path, "r") as file_descriptor:
            return get_calculation_state_from_artn_output(file_descriptor.read())
