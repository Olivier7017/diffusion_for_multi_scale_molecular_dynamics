"""Abinit single-point calculator (a ground-truth oracle for the active learning loop)."""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.calc.abinit_runner import \
    AbinitRunner
from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.io.abinit import (
    read_abinit_output, write_abinit_input)


class AbinitSinglePointCalculator(BaseSinglePointCalculator):
    """Label a single configuration with Abinit: write the input, run it, and parse the energy and forces."""

    def __init__(
        self,
        parameters: Dict,
        pseudopotentials: Dict[str, Union[str, Path]],
        abinit_runner: AbinitRunner,
        excluded_output_extensions: List[str] = ["_WFK"],
    ):
        """Init method.

        Args:
            parameters: the Abinit input variables (ASE unit convention: eV, Angstrom).
            pseudopotentials: mapping from element symbol to pseudopotential file path.
            abinit_runner: the runner that launches Abinit (serial / MPI / GPU).
            excluded_output_extensions: Abinit output files whose name contains one of these strings are
                deleted after the run.
        """
        super().__init__(self)
        self._parameters = parameters
        self._pseudopotentials = pseudopotentials
        self._abinit_runner = abinit_runner
        self._excluded_output_extensions = excluded_output_extensions
        self._calculation_type = "abinit"

    def calculate_in_work_directory(
        self, structure: Structure, work_directory: Union[Path, str]
    ) -> SinglePointCalculation:
        """Write the Abinit input, run it in work_directory, and parse the resulting energy and forces."""
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)

        atoms = structure.to_ase_atoms()
        write_abinit_input(atoms, self._parameters, self._pseudopotentials, work_directory)
        self._abinit_runner.run(work_directory)
        energy, forces, _ = read_abinit_output(work_directory)
        self._delete_excluded_outputs(work_directory)

        return SinglePointCalculation(
            calculation_type=self._calculation_type,
            structure=structure,
            forces=forces,
            energy=energy,
        )

    def _delete_excluded_outputs(self, work_directory: Path) -> None:
        """Delete large, non-essential Abinit outputs (e.g. the WFK) from the kept working directory."""
        for path in work_directory.iterdir():
            if path.is_file() and any(extension in path.name for extension in self._excluded_output_extensions):
                path.unlink()

    def calculate(self, structure: Structure, results_path: Optional[Path] = None) -> SinglePointCalculation:
        """Label a configuration with Abinit.

        Args:
            structure: the pymatgen structure to compute.
            results_path: (Optional) selects the working directory where Abinit runs and keeps all its
                outputs, as ``results_path.parent / results_path.stem``. When None, a temporary directory
                is used and discarded.

        Returns:
            calculation_results: the parsed Abinit energy and forces (forces in the input atom order).
        """
        if results_path is not None:
            results_path = Path(results_path)
            work_directory = results_path.parent / results_path.stem
            return self.calculate_in_work_directory(structure, work_directory)

        with tempfile.TemporaryDirectory() as temporary_directory:
            return self.calculate_in_work_directory(structure, Path(temporary_directory))
