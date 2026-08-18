"""Moment Tensor Potential trainer."""

from dataclasses import dataclass, field

from pymatgen.core import Element


@dataclass(kw_only=True)
class MtpConfiguration:
    """A Moment Tensor Potential model, trained with MLIP-3 and run through the lammps-mtp-kokkos interface."""

    elements: list[str]

    # Model architecture: inputs that define the potential.
    level: int  # the MTP level.
    max_dist: float  # the radial cutoff, in Angstrom.
    radial_basis_size: int  # number of radial basis functions.
    alpha_scalar_moments: int  # number of scalar moments.
    min_dist: float = 0.0  # not set by us; read back from the fitted potential after training.

    # Loss weights.
    energy_weight: float = 1.0
    force_weight: float = 1e-2
    stress_weight: float = 1e-3
    site_en_weight: float = 0.0

    # Parameters passed to the MLIP-3 'mlp train' command.
    training_params: dict = field(
        default_factory=lambda: dict(max_iter=1000, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3)
    )

    @property
    def species_count(self) -> int:
        """The number of species, derived from the elements."""
        return len(self.elements)

    @property
    def number_of_adjustable_parameters(self) -> int:
        """The number of adjustable MTP parameters."""
        return self.radial_basis_size + self.alpha_scalar_moments + self.species_count

    def __post_init__(self):
        """Validate the configuration."""
        if len(self.elements) == 0:
            raise ValueError("The list of elements should not be empty.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("The elements are not unique!")
        for element in self.elements:
            try:
                Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")

        if self.level <= 0:
            raise ValueError("The MTP level should be positive.")
        if self.max_dist <= 0.0:
            raise ValueError("The maximum distance (cutoff) should be positive.")
        if self.min_dist < 0.0:
            raise ValueError("The minimum distance should be non-negative.")
        if self.radial_basis_size <= 0:
            raise ValueError("The radial basis size should be positive.")
        if self.alpha_scalar_moments <= 0:
            raise ValueError("The number of scalar moments should be positive.")

        weights = dict(energy_weight=self.energy_weight, force_weight=self.force_weight,
                       stress_weight=self.stress_weight, site_en_weight=self.site_en_weight)
        for weight_name, weight_value in weights.items():
            if weight_value < 0.0:
                raise ValueError(f"The {weight_name} should be non-negative.")
