from __future__ import annotations
from dataclasses import dataclass

from ase.data import atomic_numbers
from ase.units import _eps0, _e, m, J
import torch
import einops

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_noisy_axl_lattice_parameters_to_unit_cell_vectors,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import (
    RepulsiveForce,
    RepulsiveForceParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, get_periodic_adjacency_information)


@dataclass(kw_only=True)
class HarmonicForceParameters(RepulsiveForceParameters):
    """Specific Hyper-parameters for the harmonic force field."""
    strength: float  # Strength of the repulsion


class HarmonicForce(RepulsiveForce):
    """Ziegler-Biersack-Littmark interatomic potential to get an analytical repulsion score."""

    def __init__(self, hyper_params: HarmonicForceParameters):
        """Initialize the ZBL analytical repulsion model which calculates forces and gives an analytical score.

        Args:
            strength : Strength of the harmonic force field
            cutoff_radius (ang): The minimal interatomic distance with no contribution from this analytical model.
            device: torch device used for internal tensors.
        """
        self._force_field_strength = hyper_params.strength
        super().__init__(hyper_params)

    def get_forces(self, A, cartesian_positions, basis_vectors) -> torch.Tensor:
        """Get relative coordinates pseudo force.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            relative_pseudo_forces : repulsive force in relative coordinates.
        """
        adj_info = get_periodic_adjacency_information(
            cartesian_positions,
            basis_vectors,
            radial_cutoff=self.cutoff_radius,
        )

        cartesian_displacements = self._get_cartesian_displacements(adj_info, cartesian_positions, basis_vectors)
        cartesian_pseudo_force_contributions = (
            self._get_cartesian_pseudo_forces_contributions(cartesian_displacements)
        )

        cartesian_pseudo_forces = self._get_cartesian_pseudo_forces(
            cartesian_pseudo_force_contributions, adj_info, cartesian_positions
        )

        reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)
        relative_pseudo_forces = get_relative_coordinates_from_cartesian_positions(
            cartesian_pseudo_forces, reciprocal_basis_vectors
        )

        return relative_pseudo_forces

    def _get_cartesian_displacements(
        self, adj_info: AdjacencyInfo, cartesian_positions, basis_vectors
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        cartesian_displacements = (
            cartesian_positions[bch, dst]
            - cartesian_positions[bch, src]
            + adj_info.shifts
        )
        return cartesian_displacements

    def _get_cartesian_pseudo_forces_contributions(self, cartesian_displacements):
        """Get cartesian pseudo forces.

        The force field is based on a potential of the form:
            phi(r) = strength * (r - cutoff_radius)^2

        The corresponding force is thus of the form
            F(r) = -nabla phi(r) = -2 strength * (r - cutoff_radius) r_hat.

        Args:
            cartesian_displacements : vectors (r_i - r_j). Dimension [number_of_edges, spatial_dimension]

        Returns:
            cartesian_pseudo_forces_contributions: Force contributions for each displacement, for the
                chosen potential. F(r_i - r_j) = - d/dr phi(r) (r_i - r_j) / ||r_i - r_j||
        """
        s = self._force_field_strength
        r0 = self.cutoff_radius

        number_of_edges, spatial_dimension = cartesian_displacements.shape

        r = torch.linalg.norm(cartesian_displacements, dim=1)

        # Add a small epsilon value in case r is close to zero, to avoid NaNs.
        epsilon = torch.tensor(1.0e-8).to(r)

        pseudo_force_prefactors = 2.0 * s * (r - r0) / (r + epsilon)
        # Repeat so we can multiply by r_hat
        repeat_pseudo_force_prefactors = einops.repeat(
            pseudo_force_prefactors, "e -> e d", d=spatial_dimension
        )
        contributions = repeat_pseudo_force_prefactors * cartesian_displacements
        return contributions

    def _get_cartesian_pseudo_forces(
        self,
        cartesian_pseudo_force_contributions: torch.Tensor,
        adj_info: AdjacencyInfo,
        cartesian_positions,
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        batch_size, natoms, spatial_dimension = cartesian_positions.shape

        # Combine the bch and src index into a single global index
        node_idx = natoms * bch + src

        list_pseudo_force_components = []

        for space_idx in range(spatial_dimension):
            pseudo_force_component = torch.zeros(natoms * batch_size).to(
                cartesian_pseudo_force_contributions
            )
            pseudo_force_component.scatter_add_(
                dim=0,
                index=node_idx,
                src=cartesian_pseudo_force_contributions[:, space_idx],
            )
            list_pseudo_force_components.append(pseudo_force_component)

        cartesian_pseudo_forces = einops.rearrange(
            list_pseudo_force_components,
            pattern="d (b n) -> b n d",
            b=batch_size,
            n=natoms,
        )
        return cartesian_pseudo_forces
