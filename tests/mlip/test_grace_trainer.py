import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_configuration import \
    GraceConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_mlip import \
    GraceMlip
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_trainer import \
    GraceTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation

MOCK_ENERGY_CENTER = -26.4  # eV; each config gets a random offset so the fit has something to learn.

# A gutted model (few descriptors) keeps the tests fast: its projection dimension is tiny, so pace_activeset
# needs only a few atomic environments and a couple of Si8 configs suffice everywhere.
TINY_MODEL_KWARGS = dict(max_order=3, lmax=[1, 1, 1], n_rad_max=[2, 2, 2],
                         embedding_size=4, fs_parameters=[[1.0, 1.0], [1.0, 0.5]])
DATABASE_SIZE = 4
SMALL_DATABASE_SIZE = 2
PRETRAINED_UPDATES = 10  # deliberately under-converged, so continuing the fit still has room to improve.
RESTART_UPDATES = 3


def create_database(reference_files_directory, number_of_configurations, perturbation=0.1, seed=0):
    """Build randomly-perturbed Si8 configurations with varied mock energies (diverse atomic environments)."""
    random_generator = np.random.default_rng(seed)
    structure_file = reference_files_directory / "structure" / "Si8.in"
    base_structure = LammpsData.from_file(str(structure_file), atom_style="atomic", sort_id=True).structure

    database = []
    for _ in range(number_of_configurations):
        structure = base_structure.copy()
        structure.perturb(perturbation)
        energy = MOCK_ENERGY_CENTER + float(random_generator.uniform(-1.0, 1.0))
        database.append(SinglePointCalculation(calculation_type="grace", structure=structure,
                                               forces=np.zeros((len(structure), 3)), energy=energy))
    return database


def build_configuration(target_total_updates):
    """A gutted single-species GRACE-FS configuration for the tests (tiny model + JIT off: fast short fits)."""
    return GraceConfiguration(elements=["Si"], size="small", cutoff=3.5, jit_compile=False,
                              model_kwargs=TINY_MODEL_KWARGS,
                              target_total_updates=target_total_updates, batch_size=4, test_batch_size=1)


def build_training_database(database):
    """Write the labelled configurations into a fresh TrainingDatabase (in a temp directory)."""
    training_database = TrainingDatabase(Path(tempfile.mkdtemp()) / "database")
    training_database.write_oracle(
        1, [calculation.to_atoms(list(range(len(calculation.structure)))) for calculation in database]
    )
    return training_database


def build_fitted_trainer(database, target_total_updates):
    """Construct a GraceTrainer over the database and fit it for target_total_updates steps (cold start)."""
    trainer = GraceTrainer(build_configuration(target_total_updates), initial_configuration=database[0],
                           training_database=build_training_database(database))
    trainer.fit()
    return trainer


def train_set_energy_rmse(model_file_path, database):
    """RMSE between the model's predicted energies and the labels over the database (via PyGRACEFSCalculator)."""
    from pyace.asecalc import PyGRACEFSCalculator
    calculator = PyGRACEFSCalculator(str(model_file_path))
    errors = []
    for calculation in database:
        atoms = calculation.structure.to_ase_atoms()
        atoms.calc = calculator
        errors.append(atoms.get_potential_energy() - calculation.energy)
    return float(np.sqrt(np.mean(np.square(errors))))


@pytest.mark.requires_grace
@pytest.mark.slow
class TestGraceRestart:
    """The full GRACE-FS restart path in one gracemaker-heavy test (two fits: the cold one, then the warm '-rl')."""

    def test_reloaded_mlip_warm_starts_from_a_cold_fitted_model(self, reference_files_directory, tmp_path):
        """Cold-fit and deploy a model, then reload it into a fresh MLIP and warm-start (-rl) a second fit.

        A single test exercises the whole restart path - the cold fit (with its pace_activeset active set),
        reloading the committed model into a fresh MLIP, and a resumed fit - and both fittings run while the
        second demonstrably lowers the train-set RMSE. The cold-fitted model is its own restart reference, so
        nothing needs to be committed to reference_files.
        """
        # 1. Cold-fit an (under-converged) model and deploy it as a committed model directory.
        reference_database = create_database(reference_files_directory, DATABASE_SIZE, seed=0)
        reference_trainer = build_fitted_trainer(reference_database, PRETRAINED_UPDATES)
        assert reference_trainer.exported_model_path.is_file()  # the cold fit produced a model

        model_directory = tmp_path / "committed_model"
        reference_trainer.write_checkpoint(model_directory)  # model.yaml + model.asi (pace_activeset) + seed/
        assert (model_directory / "model.asi").stat().st_size > 0  # the active set was built

        # 2. A restart database with room to improve, and the pre-restart error on it.
        restart_database = create_database(reference_files_directory, SMALL_DATABASE_SIZE, seed=2)
        rmse_before = train_set_energy_rmse(model_directory / "model.yaml", restart_database)

        # 3. A fresh MLIP, as a restarted process builds it: no fitted state until it loads the model.
        fresh_trainer = GraceTrainer(build_configuration(RESTART_UPDATES),
                                     initial_configuration=restart_database[0],
                                     training_database=build_training_database(restart_database))
        fresh_mlip = GraceMlip(grace_trainer=fresh_trainer, lammps_runner=MagicMock())  # toolchain on PATH
        assert not fresh_trainer._has_fitted  # a fresh MLIP would otherwise cold-start

        fresh_mlip.load(model_directory)

        assert isinstance(fresh_mlip.lammps_potential, LammpsPotential)  # the deployed potential drives the dynamics
        assert fresh_trainer._has_fitted  # the '-rl' seed was restored, so the next fit warm-starts

        # 4. The warm-started (-rl) second fit runs and lowers the error.
        fresh_trainer.fit()
        rmse_after = train_set_energy_rmse(fresh_trainer.exported_model_path, restart_database)
        assert rmse_after < rmse_before
