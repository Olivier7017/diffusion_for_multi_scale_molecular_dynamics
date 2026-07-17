"""Compute ground-truth energy and forces with the Stillinger-Weber potential in LAMMPS."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, LammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import (
    EnergyOracle, OracleParameters)


@dataclass(kw_only=True)
class LammpsOracleParameters(OracleParameters):
    """Lammps Oracle Parameters."""

    name: str = "lammps"
    sw_coeff_filename: str  # Stillinger-Weber potential filename


class LammpsEnergyOracle(EnergyOracle):
    """Lammps energy oracle.

    Batched (AXL) entry point that delegates each configuration to a Stillinger-Weber calculator.
    """

    def __init__(
        self,
        lammps_oracle_parameters: LammpsOracleParameters,
        lammps_runner: Union[LammpsRunner, InProcessLammpsRunner],
        sw_coefficients_dir: Path = SW_COEFFICIENTS_DIR,
    ):
        """Init method.

        Args:
            lammps_oracle_parameters: parameters for the LAMMPS Oracle.
            lammps_runner: a runner that executes LAMMPS (subprocess or in-process).
            sw_coefficients_dir: directory where the SW coefficient files can be found.
        """
        super().__init__(lammps_oracle_parameters)
        sw_coefficients_file_path = sw_coefficients_dir / lammps_oracle_parameters.sw_coeff_filename
        assert sw_coefficients_file_path.is_file(), \
            f"The SW file '{sw_coefficients_file_path}' does not exist."

        potential = StillingerWeberPotential(sw_coefficients_file_path=sw_coefficients_file_path)
        self._calculator = LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=lammps_runner)

    def _compute_one_configuration_energy_and_forces(
        self,
        cartesian_positions: np.ndarray,
        basis_vectors: np.ndarray,
        atom_types: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        assert np.allclose(basis_vectors, np.diag(np.diag(basis_vectors))), \
            "only orthogonal LAMMPS box are valid"

        if np.diag(basis_vectors).min() < 3.0:
            warnings.warn("Got a box with a side length smaller than 3.0 Angstrom in LAMMPS. Skipping this example.")
            return 0.0, np.zeros_like(cartesian_positions)

        species = [self._element_types.get_element(int(atom_type)) for atom_type in atom_types]
        structure = Structure(
            lattice=Lattice(matrix=basis_vectors, pbc=(True, True, True)),
            species=species,
            coords=cartesian_positions,
            coords_are_cartesian=True,
        )
        result = self._calculator.calculate(structure)
        return result.energy, result.forces
