import shutil
from pathlib import Path

import einops
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION, CARTESIAN_POSITIONS)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_energy_oracle import (
    LammpsEnergyOracle, LammpsOracleParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_number_of_lattice_parameters, map_unit_cell_to_lattice_parameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.element_types import \
    ElementTypes


class BaseTestLammpsEnergyOracle:
    """Shared fixtures and tests for the LAMMPS energy oracle. Not collected directly.

    Concrete subclasses provide the lammps_runner fixture and the availability marker.
    """

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        np.random.seed(2311331423)

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def num_atoms(self):
        return 8

    @pytest.fixture()
    def acell(self):
        return 5.5

    @pytest.fixture()
    def lattice_parameters(self, spatial_dimension, acell):
        num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
        lattice_parameters = torch.ones(num_lattice_parameters) * acell
        lattice_parameters[spatial_dimension:] = 0
        return lattice_parameters

    @pytest.fixture()
    def box(self, spatial_dimension, acell):
        return np.diag(spatial_dimension * [acell])

    @pytest.fixture()
    def cartesian_positions(self, num_atoms, spatial_dimension, box):
        x = np.random.rand(num_atoms, spatial_dimension)
        return einops.einsum(box, x, "d1 d2, natoms d2 -> natoms d1")

    @pytest.fixture()
    def number_of_unique_elements(self):
        return 2

    @pytest.fixture()
    def unique_elements(self, number_of_unique_elements):
        match number_of_unique_elements:
            case 1:
                elements = ["Si"]
            case 2:
                elements = ["Si", "Ge"]
            case _:
                raise NotImplementedError()

        return elements

    @pytest.fixture()
    def lammps_oracle_parameters(self, number_of_unique_elements, unique_elements):
        match number_of_unique_elements:
            case 1:
                sw_coeff_filename = "Si.sw"
            case 2:
                sw_coeff_filename = "SiGe.sw"
            case _:
                raise NotImplementedError()

        return LammpsOracleParameters(
            sw_coeff_filename=sw_coeff_filename, elements=unique_elements
        )

    @pytest.fixture()
    def element_types(self, unique_elements):
        return ElementTypes(unique_elements)

    @pytest.fixture()
    def atom_types(self, element_types, num_atoms):
        return np.random.choice(element_types.element_ids, num_atoms, replace=True)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def samples(self, batch_size, num_atoms, spatial_dimension, element_types):

        list_acells = 5.0 + 5.0 * torch.rand(batch_size)
        basis_vectors = torch.stack(
            [acell * torch.eye(spatial_dimension) for acell in list_acells]
        )

        relative_coordinates = torch.rand(batch_size, num_atoms, spatial_dimension)
        cartesian_positions = einops.einsum(
            basis_vectors,
            relative_coordinates,
            "batch d1 d2, batch natoms d2 -> batch natoms d1",
        )

        lattice_parameters = map_unit_cell_to_lattice_parameters(basis_vectors)
        atom_types = torch.randint(
            element_types.number_of_atom_types, (batch_size, num_atoms)
        )

        axl_composition = AXL(
            X=relative_coordinates, A=atom_types, L=lattice_parameters
        )

        return {
            CARTESIAN_POSITIONS: cartesian_positions,
            AXL_COMPOSITION: axl_composition,
        }

    @pytest.fixture()
    def oracle(self, element_types, lammps_oracle_parameters, lammps_runner):
        return LammpsEnergyOracle(lammps_oracle_parameters=lammps_oracle_parameters,
                                  lammps_runner=lammps_runner)

    def test_compute_energy_and_forces(
        self, oracle, element_types, cartesian_positions, box, atom_types
    ):
        expected_atoms = [element_types.get_element(id) for id in atom_types]
        structure = Structure(
            lattice=Lattice(matrix=box, pbc=(True, True, True)),
            species=expected_atoms,
            coords=cartesian_positions,
            coords_are_cartesian=True,
        )
        result = oracle._calculator.calculate(structure)

        # positions and elements survive the LAMMPS round-trip
        np.testing.assert_allclose(
            cartesian_positions, result.structure.cart_coords, atol=1e-5
        )
        computed_atoms = [specie.symbol for specie in result.structure.species]
        assert computed_atoms == expected_atoms

    def test_compute_oracle_energies(self, oracle, samples, batch_size):
        energies, _ = oracle.compute_oracle_energies_and_forces(samples)
        assert len(energies) == batch_size


@pytest.mark.not_on_github
@pytest.mark.requires_inprocess_lammps
class TestLammpsEnergyOracle(BaseTestLammpsEnergyOracle):
    """In-process (python binding) oracle tests over the atom/element grid."""

    @pytest.fixture(params=[8, 12, 16])
    def num_atoms(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def number_of_unique_elements(self, request):
        return request.param

    @pytest.fixture()
    def lammps_runner(self):
        return InProcessLammpsRunner()


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
class TestSubprocessLammpsEnergyOracle(BaseTestLammpsEnergyOracle):
    """Subprocess oracle tests, serial and with mpirun, at a single small configuration."""

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
