from typing import Any, AnyStr, Dict, List, Optional, Union

from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import (
    InProcessLammpsRunner, LammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import (
    EnergyOracle, OracleParameters)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_energy_oracle import (
    LammpsEnergyOracle, LammpsOracleParameters)

ORACLE_PARAMETERS_BY_NAME = dict(lammps=LammpsOracleParameters)
ENERGY_ORACLE_BY_NAME = dict(lammps=LammpsEnergyOracle)


def create_energy_oracle_parameters(
    energy_oracle_dictionary: Dict[AnyStr, Any], elements: List[str]
) -> OracleParameters:
    """Create energy oracle parameters.

    Args:
        energy_oracle_dictionary : parsed configuration for the energy oracle.
        elements : list of unique elements.

    Returns:
        oracle_parameters: a configuration object for an energy oracle object.
    """
    name = energy_oracle_dictionary["name"]

    assert (
        name in ORACLE_PARAMETERS_BY_NAME.keys()
    ), f"Energy Oracle {name} is not implemented. Possible choices are {ORACLE_PARAMETERS_BY_NAME.keys()}"

    oracle_parameters = ORACLE_PARAMETERS_BY_NAME[name](
        **energy_oracle_dictionary, elements=elements
    )
    return oracle_parameters


def create_energy_oracle(
    oracle_parameters: OracleParameters,
    lammps_runner: Optional[Union[LammpsRunner, InProcessLammpsRunner]] = None,
) -> EnergyOracle:
    """Create an energy oracle.

    This is a factory method responsible for instantiating the energy oracle. When no runner is
    provided, an in-process runner is used (the convenient default for the diffusion side).
    """
    name = oracle_parameters.name
    assert (
        name in ENERGY_ORACLE_BY_NAME.keys()
    ), f"Energy Oracle {name} is not implemented. Possible choices are {ENERGY_ORACLE_BY_NAME.keys()}"

    if lammps_runner is None:
        lammps_runner = InProcessLammpsRunner()

    oracle = ENERGY_ORACLE_BY_NAME[name](oracle_parameters, lammps_runner=lammps_runner)

    return oracle
