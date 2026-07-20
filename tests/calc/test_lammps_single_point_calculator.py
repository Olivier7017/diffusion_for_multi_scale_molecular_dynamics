import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.flare import \
    FlarePotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential

REFERENCE_FILES_DIR = Path(__file__).parent.parent / "reference_files"
STRUCTURE_FILE = REFERENCE_FILES_DIR / "structure" / "Si8.in"
MLIP_DIR = REFERENCE_FILES_DIR / "mlip"


def _stillinger_weber_potential():
    return StillingerWeberPotential(sw_coefficients_file_path=MLIP_DIR / "aSi.sw")


def _flare_potential():
    return FlarePotential(
        pair_coeff_file_path=MLIP_DIR / "flare_model.flare",
        mapped_uncertainty_file_path=MLIP_DIR / "flare_unc.flare",
    )


# (id, pair_style, potential_factory, supports_uncertainty)
POTENTIALS = [
    ("stillinger_weber", "sw", _stillinger_weber_potential, False),
    ("flare", "flare", _flare_potential, True),
]


class BaseTestLammpsSinglePointCalculator:

    @pytest.fixture()
    def structure(self):
        return LammpsData.from_file(str(STRUCTURE_FILE), atom_style="atomic", sort_id=True).structure

    def _pair_style_available(self, pair_style):
        """Whether the pair_style is available in the LAMMPS driven by this test's runner."""
        raise NotImplementedError("must be implemented in a child class.")

    @pytest.mark.parametrize("with_uncertainty", [False, True])
    @pytest.mark.parametrize(
        "potential_id, pair_style, potential_factory, supports_uncertainty",
        POTENTIALS,
        ids=[potential[0] for potential in POTENTIALS],
    )
    def test_single_point(self, lammps_runner, structure, potential_id, pair_style,
                          potential_factory, supports_uncertainty, with_uncertainty):
        if with_uncertainty and not supports_uncertainty:
            pytest.skip(f"The {potential_id} potential does not provide uncertainty.")
        if not self._pair_style_available(pair_style):
            pytest.skip(f"pair_style {pair_style} is not available in this LAMMPS.")

        calculator = LammpsSinglePointCalculator(
            potential_factory(), lammps_runner, with_uncertainty=with_uncertainty
        )
        result = calculator.calculate(structure)

        number_of_atoms = len(structure)
        assert np.ndim(result.energy) == 0 and np.isfinite(result.energy)
        assert result.forces.shape == (number_of_atoms, 3)
        assert np.all(np.isfinite(result.forces))


@pytest.mark.not_on_github
@pytest.mark.requires_inprocess_lammps
class TestInProcessLammpsSinglePointCalculator(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return InProcessLammpsRunner()

    def _pair_style_available(self, pair_style):
        import lammps

        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-echo", "none"])
        try:
            return bool(lmp.has_style("pair", pair_style))
        finally:
            lmp.close()


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
class TestSubprocessLammpsSinglePointCalculator(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture(params=["serial", "mpirun"])
    def lammps_runner(self, request):
        lammps_executable = shutil.which("lmp") or shutil.which("lammps")

        if request.param == "serial":
            return SubprocessLammpsRunner(
                lammps_executable_path=Path(lammps_executable), mpi_processors=1
            )

        if shutil.which("mpirun") is None:
            pytest.skip("No mpirun found for the SubprocessLammpsRunner.")
        return SubprocessLammpsRunner(
            lammps_executable_path=Path(lammps_executable), mpi_processors=2
        )

    def _pair_style_available(self, pair_style):
        lammps_executable = shutil.which("lmp") or shutil.which("lammps")
        output = subprocess.run(
            [lammps_executable, "-h"], capture_output=True, text=True
        ).stdout
        return pair_style in output.split()


def test_stillinger_weber_potential_rejects_uncertainty():
    potential = StillingerWeberPotential(sw_coefficients_file_path=MLIP_DIR / "aSi.sw")
    with pytest.raises(ValueError):
        potential.interaction_commands("Si", with_uncertainty=True)
