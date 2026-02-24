import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.harmonic_force import HarmonicForce, HarmonicForceParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_noisy_axl_lattice_parameters_to_unit_cell_vectors,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, get_periodic_adjacency_information)


@pytest.mark.parametrize("number_of_atoms", [4, 8, 16])
@pytest.mark.parametrize("cutoff_radius", [1.5, 2.0, 2.5])
class TestForceFieldAugmentedScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture()
    def score_network(
        self, number_of_atoms, spatial_dimension, num_atom_types
    ):
        # Generate an arbitrary MLP-based score network.
        score_network_parameters = MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            relative_coordinates_embedding_dimensions_size=6,
            noise_embedding_dimensions_size=6,
            time_embedding_dimensions_size=6,
            atom_type_embedding_dimensions_size=12,
            lattice_parameters_embedding_dimensions_size=6,
            n_hidden_dimensions=2,
            hidden_dimensions_size=16,
        )
        return MLPScoreNetwork(score_network_parameters)

    @pytest.fixture()
    def harmonic_force_parameters(self, cutoff_radius):
        return HarmonicForceParameters(cutoff_radius=cutoff_radius, strength=1.0)

    @pytest.fixture()
    def force_field_augmented_score_network(
        self, score_network, harmonic_force_parameters
    ):
        harmonic_force = HarmonicForce(harmonic_force_parameters)
        augmented_score_network = ForceFieldAugmentedScoreNetwork(
            score_network, harmonic_force
        )
        return augmented_score_network

    @pytest.fixture
    def times(self, batch_size):
        times = torch.rand(batch_size, 1)
        return times

    @pytest.fixture()
    def basis_vectors(self, batch_size, spatial_dimension):
        # orthogonal boxes with dimensions between 5 and 10.
        orthogonal_boxes = torch.stack(
            [
                torch.diag(5.0 + 5.0 * torch.rand(spatial_dimension))
                for _ in range(batch_size)
            ]
        )
        # add a bit of noise to make the vectors not quite orthogonal
        basis_vectors = orthogonal_boxes + 0.1 * torch.randn(
            batch_size, spatial_dimension, spatial_dimension
        )
        return basis_vectors

    @pytest.fixture
    def lattice_parameters(self, batch_size, spatial_dimension, basis_vectors):
        lattice_params = torch.zeros(batch_size, int(spatial_dimension * (spatial_dimension + 1) / 2))
        lattice_params[:, :spatial_dimension] = torch.diagonal(basis_vectors, dim1=-2, dim2=-1)
        return lattice_params

    @pytest.fixture
    def relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture
    def cartesian_forces(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        cartesian_forces = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return cartesian_forces

    @pytest.fixture
    def noises(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def batch(
        self,
        relative_coordinates,
        atom_types,
        cartesian_forces,
        times,
        noises,
        basis_vectors,
        lattice_parameters,
    ):
        return {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types,
                X=relative_coordinates,
                L=lattice_parameters,
            ),
            TIME: times,
            UNIT_CELL: basis_vectors,  # TODO remove this
            NOISE: noises,
            CARTESIAN_FORCES: cartesian_forces,
        }

    @pytest.fixture
    def number_of_edges(self):
        return 128

    @pytest.fixture
    def fake_cartesian_displacements(self, number_of_edges, spatial_dimension):
        return torch.rand(number_of_edges, spatial_dimension)

    def test_get_cartesian_pseudo_forces_contributions(
        self,
        force_field_augmented_score_network,
        harmonic_force_parameters,
        fake_cartesian_displacements,
    ):
        s = harmonic_force_parameters.strength
        r0 = harmonic_force_parameters.cutoff_radius

        expected_contributions = force_field_augmented_score_network._score_forces._get_cartesian_pseudo_forces_contributions(
            fake_cartesian_displacements
        )

        for r, expected_contribution in zip(
            fake_cartesian_displacements, expected_contributions
        ):
            r_norm = torch.linalg.norm(r)

            r_hat = r / r_norm
            computed_contribution = 2.0 * s * (r_norm - r0) * r_hat
            torch.testing.assert_allclose(expected_contribution, computed_contribution)

    def test_get_cartesian_pseudo_forces(
        self, batch, harmonic_force_parameters, force_field_augmented_score_network
    ):
        composition_i = batch[NOISY_AXL_COMPOSITION]
        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(composition_i.L)
        cartesian_positions = get_positions_from_coordinates(composition_i.X, basis_vectors)
        
        adj_info = get_periodic_adjacency_information(
            cartesian_positions,
            basis_vectors,
            radial_cutoff=harmonic_force_parameters.cutoff_radius,
        )
        cartesian_displacements = (
            force_field_augmented_score_network._get_cartesian_displacements(
                adj_info, batch
            )
        )
        cartesian_pseudo_force_contributions = (
            force_field_augmented_score_network._score_forces._get_cartesian_pseudo_forces_contributions(cartesian_displacements)
        )

        computed_cartesian_pseudo_forces = (
            force_field_augmented_score_network._get_cartesian_pseudo_forces(
                cartesian_pseudo_force_contributions, adj_info, batch
            )
        )

        # Compute the expected value by explicitly looping over indices, effectively checking that
        # the 'torch.scatter_add' is used correctly.
        expected_cartesian_pseudo_forces = torch.zeros_like(
            computed_cartesian_pseudo_forces
        )
        batch_indices = adj_info.edge_batch_indices
        source_indices, _ = adj_info.adjacency_matrix
        for batch_idx, src_idx, cont in zip(
            batch_indices, source_indices, cartesian_pseudo_force_contributions
        ):
            expected_cartesian_pseudo_forces[batch_idx, src_idx] += cont

        torch.testing.assert_allclose(
            computed_cartesian_pseudo_forces, expected_cartesian_pseudo_forces
        )

    def test_augmented_scores(
        self, batch, score_network, harmonic_force_parameters, force_field_augmented_score_network
    ):
        force_directions, force_importance = force_field_augmented_score_network.get_force_score_from_batch(
            batch
        )
        assert force_directions.sum() < 1e-4
       
        updated_scores = force_field_augmented_score_network(batch)
        # TODO : Add more test to make sure the updated scores make sense

        
def test_specific_scenario_sanity_check():
    """Test a specific scenario.

    It is very easy to have the forces point in the wrong direction. Here we check explicitly that
    the computed forces points AWAY from the neighbors.
    """
    spatial_dimension = 3

    harmonic_force_parameters = HarmonicForceParameters(cutoff_radius=0.4, strength=1)
    harmonic_force = HarmonicForce(harmonic_force_parameters)

    force_field_score_network = ForceFieldAugmentedScoreNetwork(
        score_network=None, score_forces=harmonic_force
    )

    # Put two atoms on a straight line
    relative_coordinates = torch.tensor([[[0.35, 0.5, 0.0], [0.65, 0.5, 0.0]]])
    atom_types = torch.zeros_like(relative_coordinates[..., 0])
    lattice_parameters = torch.ones(1, 6)
    lattice_parameters[:, 3:] = 0

    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)
    cartesian_positions = get_positions_from_coordinates(relative_coordinates, basis_vectors)

    batch = {
        NOISY_AXL_COMPOSITION: AXL(
            A=atom_types, X=relative_coordinates, L=lattice_parameters
        ),
    }

    forces = force_field_score_network._score_forces.get_forces(
        A=atom_types,
        cartesian_positions=cartesian_positions,
        basis_vectors=basis_vectors
    )

    force_on_atom1 = forces[0, 0]
    force_on_atom2 = forces[0, 1]

    assert force_on_atom1[0] < 0.0
    assert force_on_atom2[0] > 0.0

    torch.testing.assert_allclose(force_on_atom1[1:], torch.zeros(2))
    torch.testing.assert_allclose(force_on_atom2[1:], torch.zeros(2))
    torch.testing.assert_allclose(
        force_on_atom1 + force_on_atom2, torch.zeros(spatial_dimension)
    )
