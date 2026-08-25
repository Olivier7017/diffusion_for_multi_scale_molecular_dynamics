import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_configuration import \
    GraceConfiguration
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


def ensure_reference_artifacts(reference_files_directory):
    """Return (database_pkl, pretrained_directory), (re)generating them if missing.

    Generating them runs a full gracemaker fit and is slow, so they are cached under reference_files/mlip and
    reused across runs; only regenerated when absent.
    """
    # Test-specific artifacts live under a directory mirroring this test module.
    test_reference_directory = reference_files_directory / "mlip" / "grace_trainer"
    database_pkl = test_reference_directory / "grace_database.pkl.gz"
    pretrained_directory = test_reference_directory / "grace_pretrained"
    if database_pkl.exists() and (pretrained_directory / "model.yaml").exists():
        return database_pkl, pretrained_directory

    test_reference_directory.mkdir(parents=True, exist_ok=True)
    database = create_database(reference_files_directory, DATABASE_SIZE)
    trainer = build_fitted_trainer(database, PRETRAINED_UPDATES)
    trainer.write_checkpoint(pretrained_directory)  # model.yaml + model.asi + seed/
    # The training .pkl.gz written during the fit is the database pace_activeset consumes.
    shutil.copy(trainer._fit_directory / "train.pkl.gz", database_pkl)
    return database_pkl, pretrained_directory


@pytest.mark.requires_grace
@pytest.mark.slow
class TestGraceTrainer:

    def test_pace_activeset_builds_active_set(self, reference_files_directory, tmp_path):
        """pace_activeset builds a non-empty active set from the prefitted model and the premade database."""
        database_pkl, pretrained_directory = ensure_reference_artifacts(reference_files_directory)

        model_copy = tmp_path / "model.yaml"
        shutil.copy(pretrained_directory / "model.yaml", model_copy)
        subprocess.run(["pace_activeset", "-d", str(database_pkl.resolve()), str(model_copy.resolve())],
                       check=True, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        active_set_path = model_copy.with_suffix(".asi")
        assert active_set_path.is_file()
        assert active_set_path.stat().st_size > 0

    def test_restart_reduces_error(self, reference_files_directory):
        """Restarting (-rl) from the copied prefitted seed and fitting further lowers the train-set RMSE."""
        _, pretrained_directory = ensure_reference_artifacts(reference_files_directory)
        database = create_database(reference_files_directory, SMALL_DATABASE_SIZE, seed=2)
        rmse_before = train_set_energy_rmse(pretrained_directory / "model.yaml", database)

        trainer = GraceTrainer.load_checkpoint(pretrained_directory,
                                               grace_configuration=build_configuration(RESTART_UPDATES),
                                               initial_configuration=database[0],
                                               training_database=build_training_database(database))
        trainer.fit()  # resumes from the restored seed folder (-rl)

        rmse_after = train_set_energy_rmse(trainer.exported_model_path, database)
        assert rmse_after < rmse_before

    def test_cold_start_runs(self, reference_files_directory):
        """A GRACE-FS model can be fit from a cold start (no prior potential)."""
        database = create_database(reference_files_directory, SMALL_DATABASE_SIZE, seed=3)
        trainer = build_fitted_trainer(database, RESTART_UPDATES)
        assert trainer.exported_model_path.is_file()
