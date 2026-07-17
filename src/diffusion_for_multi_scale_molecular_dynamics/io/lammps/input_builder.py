"""Assemble a LAMMPS input script from composable blocks."""

from pathlib import Path
from typing import Optional, Union

from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks

DEFAULT_CONFIGURATION_FILENAME = "configuration.dat"
DUMP_FILENAME = "dump.yaml"


class LammpsInputBuilder:
    """Assemble a LAMMPS input script from six ordered blocks.

    The blocks are: initialization, configuration, potential, interruption, output and run.
    Some blocks can be empty depending on the calculations. run block could be run 0 by default.
    The build function assemble them together to make the content of the file and the
    write_lammps_input writes it to a file.
    """

    def build(
        self,
        structure: Structure,
        pair_style_command: str,
        pair_coeff_command: str,
        run_block: str,
        uncertainty_compute_command: Optional[str] = None,
        threshold: Optional[float] = None,
        initialization_block: Optional[str] = None,
        output_block: Optional[str] = None,
        configuration_filename: str = DEFAULT_CONFIGURATION_FILENAME,
    ) -> str:
        """Assemble the input script.

        Args:
            structure: the configuration (used to derive groups, masses and element symbols).
            pair_style_command: the LAMMPS pair_style command.
            pair_coeff_command: the LAMMPS pair_coeff command (elements already substituted).
            run_block: the run/driver commands (e.g. 'run 0', or the ARTn minimization block).
            uncertainty_compute_command: optional per-atom uncertainty compute (exposes 'c_unc').
            threshold: optional uncertainty threshold; when given together with an uncertainty compute,
                the interruption block is inserted.
            initialization_block: optional override for the initialization block.
            output_block: optional override for the output block.
            configuration_filename: name of the data file read by 'read_data'.

        Returns:
            input_script: the assembled LAMMPS input script.
        """
        group_block, mass_block, elements_string = generate_named_elements_blocks(structure)
        has_uncertainty = uncertainty_compute_command is not None

        blocks = [
            initialization_block if initialization_block is not None else self._initialization_block(),
            self._configuration_block(configuration_filename, group_block, mass_block),
            self._potential_block(pair_style_command, pair_coeff_command, uncertainty_compute_command),
        ]
        if has_uncertainty and threshold is not None:
            blocks.append(self._interruption_block(threshold))
        if output_block is not None:
            blocks.append(output_block)
        else:
            blocks.append(self._output_block(elements_string, has_uncertainty))
        blocks.append(run_block)

        return "\n\n\n".join(block.strip() for block in blocks if block and block.strip()) + "\n"

    def _initialization_block(self) -> str:
        return "units metal\natom_style atomic"

    def _configuration_block(self, configuration_filename: str, group_block: str, mass_block: str) -> str:
        return f"read_data {configuration_filename}\n{group_block.strip()}\n\n{mass_block.strip()}"

    def _potential_block(
        self, pair_style_command: str, pair_coeff_command: str, uncertainty_compute_command: Optional[str]
    ) -> str:
        lines = [pair_style_command, pair_coeff_command]
        if uncertainty_compute_command is not None:
            lines.append(uncertainty_compute_command)
        return "\n".join(lines)

    def _interruption_block(self, threshold: float) -> str:
        return (
            "compute max_unc_all all reduce max c_unc\n"
            "variable max_unc equal c_max_unc_all\n"
            f"variable threshold equal {threshold}\n"
            'variable continue_run equal "v_max_unc < v_threshold"\n'
            "fix extreme_extrapolation all halt 1 v_continue_run != 1"
        )

    def _output_block(self, elements_string: str, has_uncertainty: bool) -> str:
        uncertainty_field = " c_unc" if has_uncertainty else ""
        return (
            f"dump dump_id all yaml 1 {DUMP_FILENAME} id element x y z fx fy fz{uncertainty_field}\n"
            f"dump_modify dump_id element {elements_string}\n"
            "dump_modify dump_id thermo yes\n"
            "thermo 1\n"
            "thermo_style custom pe"
        )


def write_lammps_input(content: str, input_file_path: Union[Path, str]) -> None:
    """Write an assembled LAMMPS input script to disk."""
    with open(str(input_file_path), "w") as file_descriptor:
        file_descriptor.write(content)
