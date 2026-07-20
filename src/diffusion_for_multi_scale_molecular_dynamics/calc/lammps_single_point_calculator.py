import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.input_builder import (
    LammpsInputBuilder, write_lammps_input)
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class LammpsSinglePointCalculator(BaseSinglePointCalculator):
    """LAMMPS Single Point Calculator.

    Drive a single-point LAMMPS calculation for a given potential through an injected runner.
    """

    def __init__(
        self,
        lammps_potential: LammpsPotential,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        with_uncertainty: bool = False,
    ):
        """Init method.

        Args:
            lammps_potential: the potential to evaluate (emits the pair commands).
            lammps_runner: a runner that executes LAMMPS (subprocess or in-process).
            with_uncertainty: whether to compute the per-atom uncertainty.
        """
        super().__init__(self)
        self._potential = lammps_potential
        self._lammps_runner = lammps_runner
        self._with_uncertainty = with_uncertainty
        self._input_builder = LammpsInputBuilder()

        self._calculation_type = lammps_potential.calculation_type
        self._input_file_name = "lammps.in"
        self._data_filename = "configuration.dat"

    def _extract_calculation_results(self, working_directory: str) -> SinglePointCalculation:
        lammps_dump_path = Path(working_directory) / "dump.yaml"

        list_structures, list_forces, list_energies, list_uncertainties = (
            extract_all_fields(lammps_dump_path)
        )
        assert (
            len(list_structures) == 1
        ), "There is more than one frame in the dump file. This is not 'single point'!"

        result = SinglePointCalculation(
            calculation_type=self._calculation_type,
            structure=list_structures[0],
            forces=list_forces[0],
            energy=list_energies[0],
            uncertainties=list_uncertainties[0],
        )

        return result

    def _build_input(self, structure: Structure) -> str:
        """Build the LAMMPS input script for a single-point calculation."""
        return self._input_builder.build_single_point(
            structure,
            self._potential,
            with_uncertainty=self._with_uncertainty,
            configuration_filename=self._data_filename,
        )

    def calculate_in_work_directory(
        self, structure: Structure, work_directory: Union[Path, str]
    ) -> SinglePointCalculation:
        """Calculate in work directory.

        Drive LAMMPS execution in a given working directory.

        Args:
            structure: pymatgen structure.
            work_directory: work directory where inputs and outputs will be recorded.

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)

        lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
        lammps_data.write_file(str(work_directory / self._data_filename))

        input_content = self._build_input(structure)
        write_lammps_input(input_content, work_directory / self._input_file_name)

        self._lammps_runner.run_lammps(working_directory=work_directory,
                                       lammps_input_file_name=self._input_file_name)

        return self._extract_calculation_results(str(work_directory))

    def calculate(self, structure: Structure, results_path: Optional[Path] = None) -> SinglePointCalculation:
        """Calculate.

        Drive LAMMPS execution.

        Args:
            structure: pymatgen structure.
            results_path: (Optional) if present, the dump.yaml file produced by the LAMMPS calculation will
                be moved to this location.

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            calculation_result = self.calculate_in_work_directory(structure, tmp_work_dir)
            if results_path is not None:
                src = os.path.join(tmp_work_dir, "dump.yaml")
                dst = str(results_path)
                shutil.move(src, dst)

        return calculation_result
