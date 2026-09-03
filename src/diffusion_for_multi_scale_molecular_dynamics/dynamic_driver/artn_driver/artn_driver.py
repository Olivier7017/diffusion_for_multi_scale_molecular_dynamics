"""ARTn dynamic driver."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import (
    MACRO_STAGE_DESCRIPTIONS, MICRO_STAGE_DESCRIPTIONS,
    append_artn_search_summary, build_artn_lammps_tail,
    collect_artn_run_information, collect_artn_transition_information,
    get_calculation_state_from_artn_output, read_artn_search_summaries,
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
        number_of_requested_saddles: int = 1,
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
            number_of_requested_saddles: how many saddles to find (accumulated across the campaign) before the
                run reports SUCCESS; until then each run either finds saddles or is interrupted by uncertainty.
            artn_parameters: any other ARTn namelist overrides, forwarded to write_artn_input_file.
        """
        super().__init__(lammps_runner, initial_configuration)

        self._artn_library_plugin_path = self._resolve_artn_library_plugin_path(artn_library_plugin_path)
        self._push_ids = push_ids
        self._push_add_const = push_add_const
        self._number_of_requested_saddles = number_of_requested_saddles
        self._current_saddle_found = 0
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

    def summarize_interruption(self, working_directory: Path) -> List[str]:
        """Build the ARTn interrupted-run log lines from the accumulated per-search summary.

        Sums the ARTn steps and (LAMMPS-step) force evaluations over every search of this run, and takes the
        interrupting search's stage and eigenvalue from the last (interruption) record.
        """
        summaries = read_artn_search_summaries(working_directory)
        total_artn_steps = sum(record["artn_steps"] for record in summaries)
        total_force_evaluations = sum(record["force_evaluations"] or 0 for record in summaries)

        interruption = summaries[-1]
        macro_stage, micro_stage = interruption["macro_stage"], interruption["micro_stage"]
        macro_description = MACRO_STAGE_DESCRIPTIONS.get(macro_stage, macro_stage)
        micro_description = MICRO_STAGE_DESCRIPTIONS.get(micro_stage, micro_stage)
        lines = [
            f"{total_artn_steps} ARTn step, {total_force_evaluations} force evaluation.",
            f"Interrupted in {macro_stage} {micro_stage} ({macro_description}, {micro_description}).",
        ]
        eigenvalue = interruption["lowest_eigenvalue_eV_per_A2"]
        if eigenvalue is not None:  # absent until Lanczos has computed an eigenvalue (early basin steps)
            lines.append(f"lowest_eigval {eigenvalue:.4f} eV/A^2, "
                         f"eigenvector stability a1 {interruption['eigenvector_stability']:.2f}.")
        return lines

    def summarize_success(self, working_directory: Path) -> List[str]:
        """Build the ARTn success log lines: the transition's activation energies, reaction dE and locality."""
        information = collect_artn_transition_information(working_directory)
        return [
            "ARTn found a new transition pathway (1 saddle, 2 minima).",
            f"Forward activation energy {information['forward_activation_energy']:+.3f} eV, "
            f"Backward activation energy {information['backward_activation_energy']:+.3f} eV",
            f"Reaction delta Energy {information['reaction_energy']:+.3f} eV.",
            f"{information['number_of_participating_atoms']} atoms displaced by more than "
            f"{information['displacement_threshold']:g} ang.",
        ]

    def _run_dynamics(self, working_directory: Path, logger: logging.Logger) -> CalculationState:
        """Run ARTn searches until number_of_requested_saddles are found (a running total across the campaign).

        Each found saddle continues the search; an uncertainty halt returns INTERRUPTION, keeping the total so
        the next epoch resumes it; SUCCESS is reported only once the total is reached. Every search appends a
        record to the accumulated summary read back by summarize_interruption.
        """
        logger.info("Launching LAMMPS")
        total_execution_time = 0.0
        calculation_state = CalculationState.SUCCESS
        while self._current_saddle_found < self._number_of_requested_saddles:
            execution_time, succeeded = self._execute_lammps(working_directory, logger)
            total_execution_time += execution_time
            if not succeeded:
                calculation_state = CalculationState.ERROR
                break

            calculation_state = self._get_calculation_state(working_directory)
            information = collect_artn_run_information(working_directory)
            if calculation_state != CalculationState.SUCCESS:
                append_artn_search_summary(working_directory, dict(
                    outcome="interruption",
                    artn_steps=information["artn_step"],
                    force_evaluations=information["force_evaluations"],
                    macro_stage=information["macro_stage"],
                    micro_stage=information["micro_stage"],
                    lowest_eigenvalue_eV_per_A2=information["lowest_eigenvalue"],
                    eigenvector_stability=information["eigenvector_stability"],
                ))
                break

            self._current_saddle_found += 1
            barrier = collect_artn_transition_information(working_directory)["forward_activation_energy"]
            append_artn_search_summary(working_directory, dict(
                outcome="saddle",
                saddle=self._current_saddle_found,
                barrier_eV=barrier,
                artn_steps=information["artn_step"],
                force_evaluations=information["force_evaluations"],
            ))
            logger.info(f"Found saddle {self._current_saddle_found}/{self._number_of_requested_saddles}, "
                        f"barrier {barrier:.3f} eV")
        logger.info(f"Total execution time: {total_execution_time: 6.3e} sec.")
        return calculation_state

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
