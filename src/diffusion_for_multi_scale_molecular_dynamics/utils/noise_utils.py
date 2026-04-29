import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_lattice_length_scale


def get_sigma_for_relative_coordinates(
    sigma_angstrom: torch.Tensor,
    lattice_parameters: torch.Tensor,
    spatial_dimension: int,
) -> torch.Tensor:
    r"""Convert sigma from Angstrom units to relative coordinate units.

    The sigma in relative coordinates is :math:`\sigma_\text{rel} = \sigma_\text{cart} / a`, where
    :math:`a` is the cubic lattice constant. This conversion ensures that equal Cartesian sigma values
    produce the same physical noise regardless of cell size.

    Args:
        sigma_angstrom: sigma in Angstrom units. Scalar or shape [batch_size].
        lattice_parameters: Voigt lattice parameters [L11, L22, L33, ...]. Shape [batch_size, num_lattice_params].
        spatial_dimension: number of spatial dimensions.

    Returns:
        sigma_rel: sigma in relative coordinate units, shape [batch_size].
    """
    a = get_lattice_length_scale(lattice_parameters, spatial_dimension)
    return sigma_angstrom / a


def scale_sigma_by_number_of_atoms(
    sigma: torch.Tensor,
    number_of_atoms: torch.Tensor,
    spatial_dimension: int
) -> torch.Tensor:
    r"""Scale noise factor by the number of atoms.

    The variance of the noise distribution for cartesian coordinates depends on the size of the unit cell. If we assume
    the volume of a unit cell is proportional to the number of atoms, we can mitigate this variance by rescaling the
    factor :math:`\sigma` in the relative coordinates space by the number of atoms.

    .. math::

        \sigma(n) = \frac{\sigma}{\sqrt[d]n}

    with :math:`d`  the number of spatial dimensions.

    Args:
        sigma: unscaled noise factor :math:`\sigma` as a [batch_size, ...] tensor
        number_of_atoms: number of atoms in the unit cell as a [batch_size, ...] tensor
        spatial_dimension: number of spatial dimensions

    Returns:
        sigma_n : scaled sigma
    """
    return sigma / torch.pow(number_of_atoms, 1 / spatial_dimension)
