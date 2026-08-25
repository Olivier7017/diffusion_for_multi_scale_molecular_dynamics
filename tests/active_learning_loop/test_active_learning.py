"""Tests for the ActiveLearning loop orchestration and crash-resume, with stubbed stages (no LAMMPS/MLIP)."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    get_active_environment_indices
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)


def _labelled_atoms(energy: float) -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
    return atoms


def _uncertain_atoms() -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.info["uncertainty"] = np.array([0.7, 0.9])
    return atoms


def _stub_mlip():
    mlip = MagicMock()
    mlip.training_metrics.return_value = dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)
    return mlip


class RecordingActiveLearning(ActiveLearning):
    """An ActiveLearning whose three stages are stubbed to record calls and write the database artifacts.

    This exercises the real run_campaign/_run_round/resume logic without any LAMMPS or MLIP dependency.
    """

    def __init__(self, success_at_epoch: int):
        self.calls = []
        self._success_at_epoch = success_at_epoch

    def run_dynamic_driver(self, mlip, epoch):
        self.calls.append(("driver", epoch))
        if epoch >= self._success_at_epoch:
            return None  # SUCCESS: no uncertain structure found.
        return _uncertain_atoms()

    def oracle_evaluation(self, uncertain_configuration, epoch):
        self.calls.append(("oracle", epoch))
        return [_labelled_atoms(energy=float(epoch))]

    def _retrain(self, epoch, mlip, training_configurations):
        self.calls.append(("train", epoch))
        (self._training_database.model_directory(epoch) / "model").write_text("model")


def test_fresh_run_completes_and_commits_each_epoch(tmp_path):
    """A fresh run drives each epoch through all three stages until the driver reports SUCCESS."""
    active_learning = RecordingActiveLearning(success_at_epoch=3)
    active_learning.run_campaign(uncertainty_threshold=0.1, mlip=_stub_mlip(), working_directory=tmp_path)

    assert active_learning.calls == [("driver", 1), ("oracle", 1), ("train", 1),
                                     ("driver", 2), ("oracle", 2), ("train", 2),
                                     ("driver", 3)]
    database = active_learning._training_database
    for epoch in (1, 2):
        assert database.is_dynamic_committed(epoch)
        assert database.is_oracle_committed(epoch)
        assert database.is_model_committed(epoch)


def test_training_set_accumulates_across_epochs(tmp_path):
    """The database training set grows with every completed epoch (the reload-forgetting bug is gone)."""
    active_learning = RecordingActiveLearning(success_at_epoch=3)
    active_learning.run_campaign(uncertainty_threshold=0.1, mlip=_stub_mlip(), working_directory=tmp_path)

    energies = sorted(atoms.get_potential_energy() for atoms in active_learning._training_database.labelled_atoms)
    assert energies == [1.0, 2.0]


def test_resume_at_oracle_skips_the_driver(tmp_path):
    """A crash after the driver resumes at the oracle: the driver is not re-run for that epoch."""
    database = TrainingDatabase(tmp_path / "database")
    database.write_dynamic(1, _uncertain_atoms())  # epoch 1 crashed after the driver committed.

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, mlip=_stub_mlip(), working_directory=tmp_path)

    assert active_learning.calls == [("oracle", 1), ("train", 1), ("driver", 2)]


def test_resume_at_train_skips_driver_and_oracle(tmp_path):
    """A crash after labelling resumes at training: neither the driver nor the oracle re-run for that epoch."""
    database = TrainingDatabase(tmp_path / "database")
    database.write_dynamic(1, _uncertain_atoms())
    database.write_oracle(1, [_labelled_atoms(energy=1.0)])

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, mlip=_stub_mlip(), working_directory=tmp_path)

    assert active_learning.calls == [("train", 1), ("driver", 2)]


def test_forced_train_restart_retrains_a_complete_epoch(tmp_path):
    """Forcing restart_from_stage='train' on a complete epoch resets its model and retrains it."""
    database = TrainingDatabase(tmp_path / "database")
    database.write_dynamic(1, _uncertain_atoms())
    database.write_oracle(1, [_labelled_atoms(energy=1.0)])
    (database.model_directory(1) / "model").write_text("stale")  # epoch 1 was fully complete.

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, mlip=_stub_mlip(), working_directory=tmp_path,
                                 restart_from_stage="train")

    assert active_learning.calls == [("train", 1), ("driver", 2)]
    assert active_learning._training_database.is_model_committed(1)


@pytest.mark.requires_lammps_bin
@pytest.mark.requires_pair_style("sw")
class TestFullRound:
    """One real round with the actual stage internals: real Stillinger-Weber oracle + real sample maker.

    Only the driver, the LAMMPS-dump reader (covered by tests/io/lammps/test_outputs.py) and the MLIP
    training are stubbed; everything the refactor introduced (uncertainty commit, the SPC<->Atoms
    conversions, sample making, labelling, folding, the symlink) runs for real.
    """

    REFERENCE_FILES = Path(__file__).parent.parent / "reference_files"

    def test_one_real_round_wires_every_stage(self, tmp_path, monkeypatch):
        """A full round drives the uncertain config through the real oracle stage and into the model."""
        uncertain_structure = LammpsData.from_file(
            str(self.REFERENCE_FILES / "structure" / "Si8.in"), atom_style="atomic", sort_id=True
        ).structure
        uncertainty_per_atom = np.linspace(0.1, 1.0, len(uncertain_structure))

        # Real collaborators: a Stillinger-Weber oracle and a lightweight no-op sample maker.
        lammps_runner = SubprocessLammpsRunner(
            lammps_executable_path=Path(shutil.which("lmp") or shutil.which("lammps")), mpi_processors=1
        )
        oracle = LammpsSinglePointCalculator(
            StillingerWeberPotential(sw_coefficients_file_path=self.REFERENCE_FILES / "mlip" / "aSi.sw"),
            lammps_runner,
        )
        sample_maker = NoOpSampleMaker(
            sample_maker_arguments=NoOpSampleMakerArguments(element_list=["Si"]),
            atom_selector=TopKAtomSelector(TopKAtomSelectorParameters(top_k_environment=2)),
        )

        # Stubbed collaborators: the dynamic driver (INTERRUPTION then SUCCESS) and the MLIP's training.
        dynamic_driver = MagicMock()
        dynamic_driver.run.side_effect = [CalculationState.INTERRUPTION, CalculationState.SUCCESS]
        mlip = MagicMock()
        mlip.training_metrics.return_value = dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)
        mlip.train.side_effect = lambda model_directory: (Path(model_directory) / "model").write_text("model")

        active_learning = ActiveLearning(oracle_single_point_calculator=oracle,
                                         sample_maker=sample_maker, dynamic_driver=dynamic_driver)
        # Bypass only the LAMMPS-dump reader; feed the uncertain configuration directly.
        monkeypatch.setattr(
            active_learning, "_get_uncertain_structure_and_uncertainties",
            lambda working_directory, uncertainty_field: (uncertain_structure, uncertainty_per_atom),
        )

        active_learning.run_campaign(uncertainty_threshold=0.5, mlip=mlip, working_directory=tmp_path)
        database = active_learning._training_database

        # Stage 1: the uncertain configuration and its per-atom uncertainty were committed.
        assert database.is_dynamic_committed(1)
        np.testing.assert_allclose(database.read_dynamic(1).info["uncertainty"], uncertainty_per_atom)

        # Stage 2: Stillinger-Weber labelled the sample with a real energy/forces and kept the active indices.
        assert database.is_oracle_committed(1)
        labelled_atoms = database.read_oracle(1)
        assert len(labelled_atoms) == 1
        oracle_energy = labelled_atoms[0].get_potential_energy()
        assert np.isfinite(oracle_energy)
        assert labelled_atoms[0].get_forces().shape == (len(uncertain_structure), 3)
        assert "active_environment_indices" in labelled_atoms[0].info
        assert (database._root / "epoch_1" / "oracle" / "oracle_single_point_calculations.pkl").is_file()

        # Stage 3: the labelled data was folded into the model via the real SinglePointCalculation<->Atoms path.
        assert database.is_model_committed(1)
        mlip.add_labelled_structure.assert_called_once()
        single_point_calculation, active_environment_indices = mlip.add_labelled_structure.call_args.args
        np.testing.assert_allclose(single_point_calculation.energy, oracle_energy)
        assert active_environment_indices == get_active_environment_indices(labelled_atoms[0])

        # The driver ran twice (INTERRUPTION then SUCCESS) and latest_mlip points at epoch 1's model.
        assert dynamic_driver.run.call_count == 2
        assert (tmp_path / "latest_mlip").resolve() == database.model_directory(1).resolve()
