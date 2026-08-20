import re
from enum import Enum, auto


class CalculationState(Enum):
    """State of a dynamic-driver (ARTn or MD) calculation."""

    SUCCESS = auto()       # the run finished without the uncertainty exceeding the threshold.
    INTERRUPTION = auto()  # the run halted early: an uncertain structure was found.
    ERROR = auto()         # the run failed or produced unusable output.


INTERRUPTION_MESSAGE = "Failure message: ARTn RESEARCH STOP BEFORE THE END"

SUCCESS_MESSAGE = r"!> CLEANING ARTn \| Fail: 0"


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
