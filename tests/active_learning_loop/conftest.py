"""Shared fixtures for the dynamic-driver tests (base, ARTn and MD)."""

import shutil
from pathlib import Path
from typing import List, Optional

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential

# A minimal, self-contained silicon crystal used as the starting configuration.
_SILICON_LATTICE_CONSTANT = 5.43


class StubLammpsPotential(LammpsPotential):
    """A test-only potential built on a stock LAMMPS pair_style.

    It emits ``pair_style lj/cut`` (no external file needed) and uses the per-atom potential energy
    as a synthetic 'uncertainty', so the dynamic-driver machinery can be exercised end-to-end without
    any real MLIP being installed.
    """

    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return an LJ interaction section, adding the per-atom uncertainty compute on demand."""
        commands = ["pair_style lj/cut 6.0", "pair_coeff * * 0.01 2.5"]
        if with_uncertainty:
            commands.append("compute pe_atom all pe/atom")
        return commands

    def uncertainty_field(self) -> Optional[str]:
        """The per-atom compute referenced by the shared template."""
        return "c_pe_atom"


class StubMlip:
    """A stand-in for a BaseMLIP: the driver only ever reads its deployed LAMMPS potential."""

    def __init__(self, lammps_potential: LammpsPotential):
        self._lammps_potential = lammps_potential

    @property
    def lammps_potential(self) -> LammpsPotential:
        """The deployed potential."""
        return self._lammps_potential


def _write_initial_configuration(reference_directory: Path) -> Structure:
    """Write a small silicon crystal to 'initial_configuration.dat' and return its structure.

    The cell is a 2x2x2 supercell so its box (~10.9 Angstrom) comfortably exceeds the LAMMPS ghost-atom
    cutoff, which the shared template's 'comm_style tiled'/'balance' commands require.
    """
    lattice = Lattice.cubic(_SILICON_LATTICE_CONSTANT)
    structure = Structure(
        lattice=lattice,
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    structure.make_supercell([2, 2, 2])
    lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
    lammps_data.write_file(str(reference_directory / "initial_configuration.dat"))
    return structure


@pytest.fixture
def stub_potential():
    """A stub LJ-based LAMMPS potential with a synthetic uncertainty field."""
    return StubLammpsPotential()


@pytest.fixture
def stub_mlip(stub_potential):
    """A stub MLIP wrapping the stub potential."""
    return StubMlip(stub_potential)


@pytest.fixture
def reference_directory(tmp_path):
    """A reference directory holding only 'initial_configuration.dat' (for base + MD drivers)."""
    directory = tmp_path / "reference"
    directory.mkdir()
    _write_initial_configuration(directory)
    return directory


@pytest.fixture
def lammps_executable_path():
    """Path to a LAMMPS executable on PATH (only resolved for the end-to-end tests)."""
    return Path(shutil.which("lmp") or shutil.which("lammps"))
