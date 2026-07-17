"""Stillinger-Weber potential."""

from pathlib import Path

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class StillingerWeberPotential(LammpsPotential):
    """Stillinger-Weber potential."""

    calculation_type = "stillinger_weber"

    def __init__(self, sw_coefficients_file_path: Path):
        """Init method.

        Args:
            sw_coefficients_file_path: path to the Stillinger-Weber coefficient file.
        """
        self._sw_coefficients_file_path = sw_coefficients_file_path

    def pair_style_command(self) -> str:
        """Return the LAMMPS pair_style command."""
        return "pair_style sw"

    def pair_coeff_command(self, elements_string: str) -> str:
        """Return the LAMMPS pair_coeff command."""
        return f"pair_coeff * * {self._sw_coefficients_file_path} {elements_string}"
