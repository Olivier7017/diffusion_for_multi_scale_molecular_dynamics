"""Tests for the ARTn dynamic driver (unit-level generation/parsing + an end-to-end LAMMPS+ARTn run)."""

from unittest.mock import MagicMock

import pytest

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.artn_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import \
    INTERRUPTION_MESSAGE
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner

PUSH_IDS = 1
PUSH_ADD_CONST = [1.0, -1.0, -1.0, 20]


@pytest.fixture
def fake_plugin_path(tmp_path):
    """A dummy file standing in for the compiled ARTn plugin (never loaded at unit level)."""
    plugin_path = tmp_path / "libartn-lmp.so"
    plugin_path.write_text("")
    return plugin_path


@pytest.fixture
def artn_driver(reference_directory, fake_plugin_path):
    """An ARTn driver with a mock runner (unit-level: no LAMMPS is actually launched)."""
    return ArtnDriver(
        lammps_runner=MagicMock(),
        reference_directory=reference_directory,
        push_ids=PUSH_IDS,
        push_add_const=PUSH_ADD_CONST,
        artn_library_plugin_path=fake_plugin_path,
    )


class TestUnit:
    def test_prepare_reference_files_writes_artn_in(self, artn_driver, tmp_path):
        """Preparing the run generates 'artn.in' (with the push variables) in the working directory."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        artn_driver._prepare_reference_files(working_directory)

        artn_input = (working_directory / "artn.in").read_text()
        assert "&ARTN_PARAMETERS" in artn_input
        assert f"push_ids = {PUSH_IDS}" in artn_input
        assert f"push_add_const(:,{PUSH_IDS}) = 1.0, -1.0, -1.0, 20" in artn_input

    def test_dynamics_block_loads_the_plugin(self, artn_driver, fake_plugin_path):
        """The ARTn LAMMPS tail loads the plugin from the configured path."""
        assert str(fake_plugin_path) in artn_driver._dynamics_block()

    def test_missing_output_is_error(self, artn_driver, tmp_path):
        """An ARTn run that produced no 'artn.out' cannot be interpreted: ERROR."""
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.ERROR

    def test_success_message_is_success(self, artn_driver, tmp_path):
        """An 'artn.out' carrying the clean-exit message maps to SUCCESS."""
        (tmp_path / "artn.out").write_text("!> CLEANING ARTn | Fail: 0\n")
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.SUCCESS

    def test_interruption_message_is_interruption(self, artn_driver, tmp_path):
        """An 'artn.out' carrying the early-stop message maps to INTERRUPTION."""
        (tmp_path / "artn.out").write_text(f"{INTERRUPTION_MESSAGE}\n")
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.INTERRUPTION


@pytest.mark.requires_lammps_bin
class TestEndToEnd:
    def test_runs_and_returns_a_state(self, lammps_executable_path, reference_directory, stub_mlip, tmp_path):
        """A real LAMMPS+ARTn run drives to completion and yields a parseable calculation state.

        Requires a LAMMPS executable built with the ARTn plugin; the plugin location is resolved by the
        driver from the ARTN_PLUGIN_PATH environment variable (ArtnDriver raises if it is not set).
        """
        runner = SubprocessLammpsRunner(lammps_executable_path=lammps_executable_path, mpi_processors=1)
        driver = ArtnDriver(lammps_runner=runner, reference_directory=reference_directory,
                            push_ids=PUSH_IDS, push_add_const=PUSH_ADD_CONST)
        state = driver.run(stub_mlip, tmp_path / "work", uncertainty_threshold=1.0e9)
        assert isinstance(state, CalculationState)
