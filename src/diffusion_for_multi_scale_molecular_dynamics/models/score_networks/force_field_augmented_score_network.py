from dataclasses import dataclass
from typing import AnyStr, Dict, Optional

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_noisy_axl_lattice_parameters_to_unit_cell_vectors,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, get_periodic_adjacency_information)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import \
    RepulsiveForce


@dataclass(kw_only=True)
class ForceFieldParameters:
    """Force field parameters.

    The force field is based on a potential of the form:

        phi(r) = strength * (r - radial_cutoff)^2

    The corresponding force is thus of the form
        F(r) = -nabla phi(r) = -2 strength * ( r - radial_cutoff) r_hat.
    """

    radial_cutoff: float  # Cutoff to the interaction, in Angstrom
    strength: float  # Strength of the repulsion

    def __post_init__(self):
        """Post init."""
        assert (
            self.radial_cutoff > 0.0
        ), "the radial cutoff should be greater than zero."
        assert (
            self.strength > 0.0
        ), "the repulsive strength should be greater than zero."


class ForceFieldAugmentedScoreNetwork(torch.nn.Module):
    """Force Field-Augmented Score Network.

    This class wraps around an arbitrary score network in order to augment
    its output with an effective "force field". The intuition behind this is that
    atoms should never be very close to each other, but random numbers can lead
    to such proximity: a repulsive force field will encourage atoms to separate during
    diffusion.
    """

    def __init__(
        self, 
        score_network: ScoreNetwork, 
        score_forces: RepulsiveForce,
        force_activation_scale: float = 100.0,
        diffusion_time_scaling: str = "linear",
        use_for_training: bool = False,
    ):
        """Init method.

        Args:
            score_network : a score network, to be augmented with a repulsive force.
            score_forces : a repulsion_score, which will be added to the score_network
        """
        super().__init__()
        if diffusion_time_scaling != "linear":
            raise NotImplementedError(f"diffusion_time_scaling must be linear. Got {diffusion_time_scaling}")

        self._score_network = score_network
        self._score_forces = score_forces
        self._force_activation_scale = force_activation_scale
        self._diffusion_time_scaling = diffusion_time_scaling
        self._use_for_training = use_for_training
        

    def forward(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: Optional[bool] = None
    ) -> AXL:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional: if True, do a conditional forward, if False, do a unconditional forward. If None, choose
                randomly with probability conditional_prob

        Returns:
            computed_scores : the scores computed by the model.
        """
        raw_scores = self._score_network(batch, conditional)
        if self.training and not self._use_for_training:
            return raw_scores

        force_directions, force_importance = self.get_force_score_from_batch(batch)

        # Give the same norm to force_directions as raw_scores.X, with a minimum of eps if model is too small
        eps = 1e-1
        raw_norm = raw_scores.X.norm(dim=(1, 2)).clamp_min(eps)
        force_scores = force_directions * raw_norm[:, None, None]

        # Mix the two scores together, keeping raw_scores unchanged
        force_prefactor = force_importance / (1 - force_importance)
        
        updated_X_scores = raw_scores.X + force_prefactor[:, None, None] * force_scores
        updated_scores = AXL(A=raw_scores.A, X=updated_X_scores, L=raw_scores.L)
        return updated_scores

    def get_force_score_from_batch(
        self, batch: Dict[AnyStr, torch.Tensor]
    ) -> torch.Tensor:
        """Get relative coordinates repulsive score derived from _score_forces.

        The score is divided into two quantities. The normalized forces gives the direction of the score
        and the analytical fraction gives its magnitude.

        normalized_forces = F / |F|,
        where |F| is the norm over each configuration individually.

        analytical_fraction indicates how strong the correction should be as a fraction of the total score.
        It takes into account :
         1. The strength of the forces with respect to force_activation_scale.
         2. The discretization time, as atoms overlapping isn't catastrophic at t=T, but are a T=0.
        The formula linear w.r.t time (for now) :
            analytical_fraction = discretization_time * g
            g = <|F|> / (<|F|> + self.force_activation_scale)

        With this expression, the repulsion should smoothly appears when atoms become close and
        the diffusion time is getting closer to 0.

        Args:
            batch : dictionary containing the data to be processed by the model.
        
        Returns:
            normalized_forces: normalized repulsive score [Batch_size, Natoms, 3] (norm=1 for each configuration)
            analytical_fraction: repulsion score correction weight [Batch_size]
        """
        composition_i = batch[NOISY_AXL_COMPOSITION]
        time = batch[TIME]

        epsilon = 1e-12  # So force doesn't diverge if every atom is farther than cutoff_radius
        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(composition_i.L)
        cartesian_positions = get_positions_from_coordinates(composition_i.X, basis_vectors)

        forces = self._score_forces.get_forces(composition_i.A, cartesian_positions, basis_vectors)
        normalization_over_batch = forces.norm(dim=[1, 2]).clamp_min(epsilon)  # [B]
        normalized_forces = forces / normalization_over_batch[:, None, None]  # [B,N,3]

        g = normalization_over_batch / (normalization_over_batch + self._force_activation_scale)  # [B]
        analytical_fraction = (1-time[:, 0]) * g[:]  # [B] -> Unsure why time.shape = (B, 1) instead of (B)

        return normalized_forces, analytical_fraction
