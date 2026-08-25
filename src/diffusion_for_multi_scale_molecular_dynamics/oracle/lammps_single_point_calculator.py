import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.single_point_calc_lammps_input import (
    LammpsInputBuilder, write_lammps_input)
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)


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

    def _extract_calculation_results(
        self, working_directory: str, dump_filename: str = "dump.yaml"
    ) -> SinglePointCalculation:
        lammps_dump_path = Path(working_directory) / dump_filename

        list_structures, list_forces, list_energies, list_uncertainties = (
            extract_all_fields(lammps_dump_path, uncertainty_field=self._potential.uncertainty_field())
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

    def calculate_many_in_work_directory(
        self, structures: List[Structure], work_directory: Union[Path, str]
    ) -> List[SinglePointCalculation]:
        """Evaluate several configurations with a single LAMMPS run in a given working directory.

        A single looping input reads one data file and writes one dump file per configuration, so LAMMPS is
        launched only once (one process for the subprocess runner, one instance for the in-process runner).

        Args:
            structures: the configurations to evaluate, in order.
            work_directory: work directory where inputs and outputs will be recorded.

        Returns:
            calculation_results: the parsed LAMMPS output, one per structure (in the same order).
        """
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)

        configuration_filenames = [f"configuration_{index}.dat" for index in range(len(structures))]
        dump_filenames = [f"dump_{index}.yaml" for index in range(len(structures))]

        for structure, configuration_filename in zip(structures, configuration_filenames):
            lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
            lammps_data.write_file(str(work_directory / configuration_filename))

        input_content = self._input_builder.build_looping_single_point(
            structures, self._potential, configuration_filenames, dump_filenames,
            with_uncertainty=self._with_uncertainty,
        )
        write_lammps_input(input_content, work_directory / self._input_file_name)

        self._lammps_runner.run_lammps(working_directory=work_directory,
                                       lammps_input_file_name=self._input_file_name)

        return [self._extract_calculation_results(str(work_directory), dump_filename)
                for dump_filename in dump_filenames]

    def calculate_many(self, structures: List[Structure]) -> List[SinglePointCalculation]:
        """Evaluate several configurations with a single LAMMPS run (see ``calculate_many_in_work_directory``)."""
        if not structures:
            return []
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            return self.calculate_many_in_work_directory(structures, tmp_work_dir)
