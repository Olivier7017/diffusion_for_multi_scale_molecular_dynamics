import numpy as np
import pymatgen
import pytest

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import (
    generate_named_elements_blocks, sort_elements_by_atomic_mass)
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.single_point_calc_lammps_input import \
    LammpsInputBuilder
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR


@pytest.fixture()
def expected_group_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        lines += f"\ngroup {symbol} type {group_id}"

    return lines


@pytest.fixture()
def expected_mass_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        mass = pymatgen.core.Element(symbol).atomic_mass.real
        lines += f"\nmass {group_id} {mass}"
    return lines


@pytest.fixture()
def expected_elements_string(expected_sorted_list_element_symbols):
    return " ".join(expected_sorted_list_element_symbols)


def test_sort_elements_by_atomic_mass(list_elements, expected_sorted_list_element_symbols):
    computed_sorted_list_elements = sort_elements_by_atomic_mass(list_elements)
    computed_sorted_list_element_symbols = [element.symbol for element in computed_sorted_list_elements]
    assert computed_sorted_list_element_symbols == expected_sorted_list_element_symbols


def test_generate_named_elements_blocks(structure, expected_group_block, expected_mass_block, expected_elements_string):
    group_block, mass_block, elements_string = generate_named_elements_blocks(structure)
    assert group_block == expected_group_block
    assert mass_block == expected_mass_block
    assert elements_string == expected_elements_string


def test_build_looping_single_point(structure):
    """A looping input holds one self-contained block per structure, separated by 'clear'."""
    potential = StillingerWeberPotential(sw_coefficients_file_path=SW_COEFFICIENTS_DIR / "Si.sw")
    structures = [structure, structure, structure]
    configuration_filenames = [f"configuration_{index}.dat" for index in range(len(structures))]
    dump_filenames = [f"dump_{index}.yaml" for index in range(len(structures))]

    input_script = LammpsInputBuilder().build_looping_single_point(
        structures, potential, configuration_filenames, dump_filenames
    )

    # One 'clear' separates each pair of blocks; one read_data / run 0 / dump per structure.
    assert input_script.count("clear") == len(structures) - 1
    assert input_script.count("read_data") == len(structures)
    assert input_script.count("run 0") == len(structures)
    for configuration_filename, dump_filename in zip(configuration_filenames, dump_filenames):
        assert f"read_data {configuration_filename}" in input_script
        assert dump_filename in input_script
