import shutil
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
FLARE_SGP_FILE = MLIP_DIR / "flare_sgp.json"
FLARE_PAIR_COEFF_FILE = MLIP_DIR / "lammps_flare.flare"
FLARE_MAPPED_UNCERTAINTY_FILE = MLIP_DIR / "mapped_unc_flare.flare"


def _stillinger_weber_potential():
    return StillingerWeberPotential(sw_coefficients_file_path=MLIP_DIR / "aSi.sw")


def _flare_potential():
    return FlarePotential(
        pair_coeff_file_path=FLARE_PAIR_COEFF_FILE,
        mapped_uncertainty_file_path=FLARE_MAPPED_UNCERTAINTY_FILE,
    )


def _lammps_executable_path():
    return Path(shutil.which("lmp") or shutil.which("lammps"))


# (potential_id, potential_factory, supports_uncertainty); each case requires its pair_style in LAMMPS.
POTENTIALS = [
    pytest.param("stillinger_weber", _stillinger_weber_potential, False,
                 marks=pytest.mark.requires_pair_style("sw"), id="stillinger_weber"),
    pytest.param("flare", _flare_potential, True,
                 marks=pytest.mark.requires_pair_style("flare"), id="flare"),
]


class BaseTestLammpsSinglePointCalculator:

    @pytest.fixture()
    def structure(self):
        return LammpsData.from_file(str(STRUCTURE_FILE), atom_style="atomic", sort_id=True).structure

    @pytest.mark.parametrize("potential_id, potential_factory, supports_uncertainty", POTENTIALS)
    def test_single_point(self, lammps_runner, structure, potential_id, potential_factory, supports_uncertainty):
        """A single-point calculation returns finite energy and forces (and uncertainty when supported)."""
        calculator = LammpsSinglePointCalculator(
            potential_factory(), lammps_runner, with_uncertainty=supports_uncertainty
        )
        result = calculator.calculate(structure)

        number_of_atoms = len(structure)
        assert np.ndim(result.energy) == 0 and np.isfinite(result.energy)
        assert result.forces.shape == (number_of_atoms, 3)
        assert np.all(np.isfinite(result.forces))

        if supports_uncertainty:
            assert result.uncertainties.shape == (number_of_atoms,)
            assert np.all(np.isfinite(result.uncertainties))

    @pytest.mark.requires_flare
    @pytest.mark.requires_pair_style("flare")
    def test_mapped_flare_matches_sgp(self, lammps_runner, structure):
        """The mapped FLARE potential run through LAMMPS agrees with its source SGP on energy and forces."""
        import json

        from flare.bffs.sgp import SGP_Wrapper

        from diffusion_for_multi_scale_molecular_dynamics.calc.flare_single_point_calculator import \
            FlareSinglePointCalculator

        with open(FLARE_SGP_FILE) as file:
            sgp_model, kernels = SGP_Wrapper.from_dict(json.load(file))
        assert kernels  # keep the kernels alive to avoid a garbage-collection segfault in the C++ backend
        sgp_calculator = FlareSinglePointCalculator(sgp_model)

        lammps_calculator = LammpsSinglePointCalculator(_flare_potential(), lammps_runner)

        lammps_result = lammps_calculator.calculate(structure)
        sgp_result = sgp_calculator.calculate(structure)

        np.testing.assert_allclose(lammps_result.energy, sgp_result.energy, atol=1e-1)
        np.testing.assert_allclose(lammps_result.forces, sgp_result.forces, atol=1e-1)


@pytest.mark.not_on_github
@pytest.mark.requires_inprocess_lammps
class TestInProcess(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return InProcessLammpsRunner()


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
class TestSubprocessSerial(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return SubprocessLammpsRunner(lammps_executable_path=_lammps_executable_path(), mpi_processors=1)


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
@pytest.mark.requires_mpirun
class TestSubprocessMpi(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return SubprocessLammpsRunner(lammps_executable_path=_lammps_executable_path(), mpi_processors=2)


def test_stillinger_weber_potential_rejects_uncertainty():
    potential = StillingerWeberPotential(sw_coefficients_file_path=MLIP_DIR / "aSi.sw")
    with pytest.raises(ValueError):
        potential.interaction_commands("Si", with_uncertainty=True)
