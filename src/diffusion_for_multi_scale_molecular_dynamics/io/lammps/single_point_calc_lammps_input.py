"""Assemble a LAMMPS input script from composable blocks."""

from pathlib import Path
from typing import List, Union

from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential

DEFAULT_CONFIGURATION_FILENAME = "configuration.dat"
DUMP_FILENAME = "dump.yaml"


class LammpsInputBuilder:
    """Assemble a LAMMPS input script from six ordered blocks.

    The blocks are: initialization, configuration, potential, interruption, output and run.
    Some blocks can be empty depending on the calculations. run block could be run 0 by default.
    The build function assemble them together to make the content of the file and the
    write_lammps_input writes it to a file.
    """

    def build_single_point(
        self,
        structure: Structure,
        potential: LammpsPotential,
        with_uncertainty: bool = False,
        configuration_filename: str = DEFAULT_CONFIGURATION_FILENAME,
    ) -> str:
        """Assemble the LAMMPS input for a single-point calculation.

        Args:
            structure: the configuration to evaluate.
            potential: the potential providing the interaction commands and dump fields.
            with_uncertainty: whether to compute the per-atom uncertainty.
            configuration_filename: name of the data file read by 'read_data'.

        Returns:
            input_script: the assembled LAMMPS input script.
        """
        group_block, mass_block, elements_string = generate_named_elements_blocks(structure)

        blocks = [
            self._initialization_block(),
            self._configuration_block(configuration_filename, group_block, mass_block),
            "\n".join(potential.interaction_commands(elements_string, with_uncertainty=with_uncertainty)),
            self._output_block(elements_string, potential.dump_fields(with_uncertainty=with_uncertainty)),
            "run 0",
        ]

        return "\n\n\n".join(block.strip() for block in blocks if block and block.strip()) + "\n"

    def _initialization_block(self) -> str:
        return "units metal\natom_style atomic"

    def _configuration_block(self, configuration_filename: str, group_block: str, mass_block: str) -> str:
        return f"read_data {configuration_filename}\n{group_block.strip()}\n\n{mass_block.strip()}"

    def _output_block(self, elements_string: str, dump_fields: List[str]) -> str:
        fields = " ".join(dump_fields)
        return (
            f"dump dump_id all yaml 1 {DUMP_FILENAME} {fields}\n"
            f"dump_modify dump_id element {elements_string}\n"
            "dump_modify dump_id sort id\n"
            "dump_modify dump_id thermo yes\n"
            "thermo 1\n"
            "thermo_style custom pe"
        )


def write_lammps_input(content: str, input_file_path: Union[Path, str]) -> None:
    """Write an assembled LAMMPS input script to disk."""
    with open(str(input_file_path), "w") as file_descriptor:
        file_descriptor.write(content)
