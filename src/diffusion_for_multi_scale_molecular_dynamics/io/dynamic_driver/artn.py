"""ARTn input writing, LAMMPS-tail building, and output parsing.

The ARTn 'artn.in' file (a Fortran ``&ARTN_PARAMETERS`` namelist) and the ARTn LAMMPS commands are
generated here from parameters, so no template file or user-supplied artn.in is needed.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState

INTERRUPTION_MESSAGE = "Failure message: ARTn RESEARCH STOP BEFORE THE END"
SUCCESS_MESSAGE = r"!> CLEANING ARTn \| Fail: 0"

# Typical ARTn method variables; override any of them through the write_artn_input_file 'artn_parameters'.
DEFAULT_ARTN_PARAMETERS = {
    "engine_units": "lammps/metal",
    "verbose": 2,
    "ninit": 2,
    "lpush_final": True,
    "nsmooth": 2,
    "forc_thr": 0.01,
    "push_step_size": 0.1,
    "push_mode": "list",
    "lanczos_disp": 1e-4,
    "lanczos_max_size": 10,
    "lanczos_min_size": 3,
    "lanczos_eval_conv_thr": 1e-2,
    "eigen_step_size": 0.1,
    "push_over": 2.0,
}


def _format_namelist_value(value) -> str:
    """Format a Python value as a Fortran namelist value (bool, string, list, or number)."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_namelist_value(item) for item in value)
    return str(value)


def write_artn_input_file(
    path: Union[str, Path],
    push_ids: int,
    push_add_const: List[float],
    artn_parameters: Optional[Dict] = None,
) -> Path:
    """Write the ARTn '&ARTN_PARAMETERS' namelist to path, returning the written path.

    push_ids selects the atom ARTn pushes to escape the initial basin, and push_add_const is its
    four-component push constraint (written as the Fortran array line 'push_add_const(:,<push_ids>) = ...').
    artn_parameters overrides DEFAULT_ARTN_PARAMETERS and may add any other ARTn variable (see the
    ARTn documentation for the variables).
    """
    namelist = dict(DEFAULT_ARTN_PARAMETERS)
    if artn_parameters is not None:
        namelist.update(artn_parameters)

    namelist["push_ids"] = push_ids
    namelist[f"push_add_const(:,{push_ids})"] = push_add_const

    lines = ["&ARTN_PARAMETERS"]
    lines += [f"  {key} = {_format_namelist_value(value)}" for key, value in namelist.items()]
    lines += ["/"]

    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def build_artn_lammps_tail(artn_library_plugin_path: Union[str, Path]) -> str:
    """Build the ARTn LAMMPS commands: load the plugin, add the ARTn fix, and run a FIRE minimization."""
    return "\n".join([
        f"plugin load {artn_library_plugin_path}",
        "fix artn_fix_id all artn dmax 5.0",
        "timestep 0.001",
        "reset_timestep 0",
        "min_style fire",
        "minimize 1e-4 1e-5 5000 10000",
    ])


def get_calculation_state_from_artn_output(artn_output: str) -> CalculationState:
    """Get calculation state from ARTn output.

    This method determines if the ARTn calculation was successful or interrupted
    by seeking well defined sub-strings in the file content.

    Args:
        artn_output (str): The content of an artn.out file


    Returns:
        state: the parsed status of the calculation.
    """
    match_success = re.search(SUCCESS_MESSAGE, artn_output)
    match_interruption = re.search(INTERRUPTION_MESSAGE, artn_output)

    if match_success and match_interruption:
        raise ValueError("Both the success and the interruption messages are present in the artn.out file. "
                         "Something is wrong; review code!")

    if not match_success and not match_interruption:
        # The run produced an artn.out that reports neither outcome: treat it as a failed run.
        return CalculationState.ERROR

    if match_interruption:
        return CalculationState.INTERRUPTION
    else:
        return CalculationState.SUCCESS


def get_saddle_energy(artn_output: str):
    """Get saddle energy from ARTn output."""
    saddle_energy_pattern = r"\|> DEBRIEF\(SADDLE\) \| dE = (?P<energy>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) eV"
    match = re.search(saddle_energy_pattern, artn_output)
    return float(match.group('energy'))
