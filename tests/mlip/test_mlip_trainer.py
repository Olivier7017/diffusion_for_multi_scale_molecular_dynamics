import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential

# A fictitious energy label (eV); its actual value is irrelevant, we only check the fit reproduces it.
MOCK_ENERGY = -26.43783


def predict_energy(trainer_type, trainer, structure):
    """Predict the energy of a structure with the trainer's fitted model."""
    if trainer_type == "flare":
        from diffusion_for_multi_scale_molecular_dynamics.calc.flare_single_point_calculator import \
            FlareSinglePointCalculator
        return FlareSinglePointCalculator(trainer.sgp_model).calculate(structure).energy
    raise ValueError(f"Unknown trainer type '{trainer_type}'.")


class TestMLIPTrainer:

    @pytest.fixture(params=[pytest.param("flare", marks=pytest.mark.requires_flare)])
    def trainer_type(self, request):
        return request.param

    @pytest.fixture
    def trainer(self, trainer_type, list_element_symbols):
        if trainer_type == "flare":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import (
                FlareConfiguration, FlareTrainer)
            return FlareTrainer(FlareConfiguration(cutoff=5.0,
                                                   elements=list_element_symbols,
                                                   n_radial=8,
                                                   lmax=3,
                                                   variance_type='local',
                                                   initial_sigma_e=1e-8))
        raise ValueError(f"Unknown trainer type '{trainer_type}'.")

    @pytest.fixture
    def labelled_structure(self, structure):
        # We drop the forces (train on energy only) so the fitted model can reproduce its label.
        number_of_atoms = len(structure)
        return SinglePointCalculation(calculation_type='dummy_test',
                                      structure=structure,
                                      forces=np.zeros((number_of_atoms, 3)),
                                      energy=MOCK_ENERGY)

    @pytest.fixture
    def active_environment_indices(self, structure):
        return list(np.arange(len(structure)))

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
        """After fitting on a single structure, the model reproduces its mock training energy within 1e-1."""
        trainer.add_labelled_structure(labelled_structure, active_environment_indices)
        trainer.fit()

        predicted_energy = predict_energy(trainer_type, trainer, labelled_structure.structure)
        np.testing.assert_allclose(predicted_energy, labelled_structure.energy, atol=1e-1)

    def test_checkpoint_and_write_lammps_potential(self, trained_trainer, tmp_path):
        """A trained trainer's checkpoint round-trips to an identical checkpoint, and it deploys a LAMMPS potential."""
        # Checkpoint the trained model before deploying: build_map (in write_lammps_potential) mutates the
        # SGP by attaching a variance kernel, which makes a post-deploy checkpoint non-idempotent on reload.
        checkpoint_path = tmp_path / "checkpoint.json"
        trained_trainer.write_checkpoint(checkpoint_path)
        assert checkpoint_path.is_file()

        reloaded_checkpoint_path = tmp_path / "reloaded_checkpoint.json"
        type(trained_trainer).load_checkpoint(checkpoint_path).write_checkpoint(reloaded_checkpoint_path)
        assert reloaded_checkpoint_path.read_text() == checkpoint_path.read_text()

        potential = trained_trainer.write_lammps_potential(tmp_path / "deploy")
        assert isinstance(potential, LammpsPotential)
        assert len(list((tmp_path / "deploy").iterdir())) > 0
