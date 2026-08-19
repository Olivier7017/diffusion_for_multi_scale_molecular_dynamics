from pathlib import Path

import numpy as np
import pytest
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential

# A fictitious energy label (eV); its actual value is irrelevant, we only check the fit reproduces it.
MOCK_ENERGY = -26.43783

SI8_STRUCTURE_FILE = Path(__file__).parent.parent / "reference_files" / "structure" / "Si8.in"

# Parameters read back from a level-6 MTP fitted on the fixed Si8 structure. The radial basis size, alpha scalar
# moments and species count are fixed by the level; min_dist is the (deterministic) minimum interatomic distance.
MTP_LEVEL = 6
MTP_MAX_DIST = 3.5
MTP_EXPECTED_RADIAL_BASIS_SIZE = 8
MTP_EXPECTED_ALPHA_SCALAR_MOMENTS = 5
MTP_EXPECTED_SPECIES_COUNT = 1
MTP_EXPECTED_MIN_DIST = 2.3336


def verify_fitted_model(trainer_type, trainer, labelled_structure):
    """Assert the fitted model matches its label (flare: predicted energy; mtp: golden fitted parameters)."""
    if trainer_type == "flare":
        from diffusion_for_multi_scale_molecular_dynamics.calc.flare_single_point_calculator import \
            FlareSinglePointCalculator
        predicted_energy = FlareSinglePointCalculator(trainer.sgp_model).calculate(labelled_structure.structure).energy
        np.testing.assert_allclose(predicted_energy, labelled_structure.energy, atol=1e-1)
    elif trainer_type == "mtp":
        configuration = trainer.configuration
        assert configuration.radial_basis_size == MTP_EXPECTED_RADIAL_BASIS_SIZE
        assert configuration.alpha_scalar_moments == MTP_EXPECTED_ALPHA_SCALAR_MOMENTS
        assert configuration.species_count == MTP_EXPECTED_SPECIES_COUNT
        assert configuration.number_of_adjustable_parameters == 14
        np.testing.assert_allclose(configuration.min_dist, MTP_EXPECTED_MIN_DIST, atol=1e-3)
    else:
        raise ValueError(f"Unknown trainer type '{trainer_type}'.")


def checkpoint_file_name(trainer_type):
    """The file write_checkpoint produces for each trainer type."""
    return {"flare": "checkpoint.json", "mtp": "potential.almtp"}[trainer_type]


def reload_trainer(trainer_type, trainer, checkpoint_path, elements):
    """Reconstruct a trainer from a checkpoint (MTP needs the configuration, which the .almtp does not record)."""
    if trainer_type == "flare":
        return type(trainer).load_checkpoint(checkpoint_path)
    elif trainer_type == "mtp":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
            MtpConfiguration
        configuration = MtpConfiguration(elements=elements, level=MTP_LEVEL, max_dist=MTP_MAX_DIST)
        return type(trainer).load_checkpoint(checkpoint_path, mtp_configuration=configuration)
    raise ValueError(f"Unknown trainer type '{trainer_type}'.")


class TestMLIPTrainer:

    @pytest.fixture(
        params=[
            pytest.param("flare", marks=pytest.mark.requires_flare),
            pytest.param("mtp", marks=pytest.mark.requires_mlp),
        ]
    )
    def trainer_type(self, request):
        return request.param

    @pytest.fixture
    def elements(self, trainer_type, list_element_symbols):
        # The level-6 MTP template is single-species; flare handles the shared multi-species set.
        return ["Si"] if trainer_type == "mtp" else list_element_symbols

    @pytest.fixture
    def training_structure(self, trainer_type, structure):
        # MTP needs a fixed single-species structure so the fitted parameters are deterministic.
        if trainer_type == "mtp":
            return LammpsData.from_file(str(SI8_STRUCTURE_FILE), atom_style="atomic", sort_id=True).structure
        return structure

    @pytest.fixture
    def trainer(self, trainer_type, elements):
        if trainer_type == "flare":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import (
                FlareConfiguration, FlareTrainer)
            return FlareTrainer(FlareConfiguration(cutoff=4.0,
                                                   elements=elements,
                                                   n_radial=4,
                                                   lmax=2,
                                                   variance_type='local',
                                                   initial_sigma_e=1e-8))
        if trainer_type == "mtp":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import (
                MtpConfiguration, MtpTrainer)
            configuration = MtpConfiguration(
                elements=elements,
                level=MTP_LEVEL,
                max_dist=MTP_MAX_DIST,
                training_params=dict(max_iter=100, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3),
            )
            return MtpTrainer(configuration)
        raise ValueError(f"Unknown trainer type '{trainer_type}'.")

    @pytest.fixture
    def labelled_structure(self, training_structure):
        # We drop the forces (train on energy only) so the fitted model can reproduce its label.
        number_of_atoms = len(training_structure)
        return SinglePointCalculation(calculation_type='dummy_test',
                                      structure=training_structure,
                                      forces=np.zeros((number_of_atoms, 3)),
                                      energy=MOCK_ENERGY)

    @pytest.fixture
    def active_environment_indices(self, training_structure):
        return list(np.arange(len(training_structure)))

    @pytest.fixture
    def trained_trainer(self, trainer, labelled_structure, active_environment_indices):
        trainer.add_labelled_structure(labelled_structure, active_environment_indices)
        trainer.fit()
        return trainer

    def test_add_labelled_structure(self, trainer, labelled_structure, active_environment_indices):
        """Adding a labelled structure appends it to the trainer's training database."""
        assert len(trainer.labelled_calculations) == 0
        trainer.add_labelled_structure(labelled_structure, active_environment_indices)
        assert len(trainer.labelled_calculations) == 1
        assert trainer.labelled_calculations[0] is labelled_structure

    def test_fit(self, trainer_type, trainer, labelled_structure, active_environment_indices):
        """After fitting on a single structure, the fitted model matches its label."""
        trainer.add_labelled_structure(labelled_structure, active_environment_indices)
        trainer.fit()
        verify_fitted_model(trainer_type, trainer, labelled_structure)

    def test_write_checkpoint(self, trainer_type, trained_trainer, elements, tmp_path):
        """write_checkpoint deploys a LAMMPS potential and round-trips to an identical checkpoint."""
        potential = trained_trainer.write_checkpoint(tmp_path / "first")
        assert isinstance(potential, LammpsPotential)

        checkpoint_path = tmp_path / "first" / checkpoint_file_name(trainer_type)
        assert checkpoint_path.is_file()

        # The checkpoint is written before any model-mutating step, so reloading and re-writing it is identical.
        reloaded_trainer = reload_trainer(trainer_type, trained_trainer, checkpoint_path, elements)
        reloaded_trainer.write_checkpoint(tmp_path / "second")
        second_checkpoint_path = tmp_path / "second" / checkpoint_file_name(trainer_type)
        assert second_checkpoint_path.read_bytes() == checkpoint_path.read_bytes()
