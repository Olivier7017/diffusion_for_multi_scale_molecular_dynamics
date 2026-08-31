"""Read a real LAMMPS text dump (a committed reference file) and check every field is recovered.

The reference dump was produced by LAMMPS for a genuinely triclinic (60-degree) 16-atom silicon cell, so
it guards the ase-based reader against the box being misread as orthogonal.
"""

from pathlib import Path

import numpy as np
from ase.build import bulk

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields_from_dump
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_pymatgen_structure

REFERENCE_DIRECTORY = Path(__file__).parents[2] / "reference_files" / "lammps"
TRICLINIC_DUMP = REFERENCE_DIRECTORY / "triclinic_single_point.dump"
ORTHOGONAL_DUMP = REFERENCE_DIRECTORY / "orthogonal_single_point.dump"


def _expected_triclinic_structure():
    """The exact structure the triclinic reference dump was generated from."""
    return to_pymatgen_structure(bulk("Si", "diamond", a=5.43).repeat((2, 2, 2)))


def test_reader_recovers_the_triclinic_structure():
    """The dump reads back to the known triclinic structure: cell, positions and species all match."""
    expected_structure = _expected_triclinic_structure()

    list_structures, _, _ = extract_all_fields_from_dump(TRICLINIC_DUMP)

    assert len(list_structures) == 1
    structure = list_structures[0]

    # The box is triclinic (all angles 60 degrees) and must be read as such, not collapsed to orthogonal.
    np.testing.assert_allclose(structure.lattice.angles, 60.0, atol=1e-4)
    np.testing.assert_allclose(structure.lattice.parameters, expected_structure.lattice.parameters, atol=1e-4)

    # Same atoms: species and (minimum-image) fractional coordinates.
    assert [str(site.specie) for site in structure.sites] == ["Si"] * len(expected_structure)
    fractional_difference = (structure.frac_coords - expected_structure.frac_coords + 0.5) % 1.0 - 0.5
    np.testing.assert_allclose(fractional_difference, 0.0, atol=1e-4)


def test_reader_recovers_forces_and_uncertainties():
    """Forces and the per-atom uncertainty column are read with the right shape and finite values."""
    number_of_atoms = len(_expected_triclinic_structure())
    _, list_forces, list_uncertainties = extract_all_fields_from_dump(TRICLINIC_DUMP)

    assert list_forces[0].shape == (number_of_atoms, 3)
    assert np.all(np.isfinite(list_forces[0]))

    assert list_uncertainties[0].shape == (number_of_atoms,)
    assert np.all(np.isfinite(list_uncertainties[0]))


def test_reader_reads_the_expected_per_atom_uncertainties():
    """The per-atom uncertainty column is read value-for-value (the fixture set c_unc_at to the atom id)."""
    _, _, list_uncertainties = extract_all_fields_from_dump(ORTHOGONAL_DUMP)

    number_of_atoms = 8  # the orthogonal reference cell has 8 atoms
    expected_uncertainties = np.arange(1, number_of_atoms + 1)  # c_unc_at was set to the atom id
    np.testing.assert_allclose(list_uncertainties[0], expected_uncertainties)


def test_reader_returns_none_uncertainty_when_field_absent():
    """Requesting a per-atom column the dump does not contain yields None (not an error)."""
    _, _, list_uncertainties = extract_all_fields_from_dump(TRICLINIC_DUMP, uncertainty_field="c_not_a_field")
    assert list_uncertainties[0] is None
