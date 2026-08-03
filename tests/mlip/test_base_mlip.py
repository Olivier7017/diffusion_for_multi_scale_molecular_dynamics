from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP


def make_calculation(energy, forces):
    return SinglePointCalculation(calculation_type="dummy_test",
                                  structure=MagicMock(),
                                  forces=np.asarray(forces, dtype=float),
                                  energy=energy)


class ConcreteMLIP(BaseMLIP):
    """Minimal concrete MLIP to exercise the base class's concrete methods with a mocked trainer."""

    def train(self, output_directory):
        pass

    def write_state_yaml(self, output_path):
        pass

    def write_logger_info(self, logger):
        pass

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        pass


class TestBaseMLIP:

    @pytest.fixture
    def trainer(self):
        return MagicMock()

    @pytest.fixture
    def mlip(self, trainer):
        return ConcreteMLIP(trainer=trainer, lammps_runner=MagicMock())

    def test_training_metrics_without_data(self, mlip, trainer):
        """With no training configuration, metrics report zero configurations and no RMSE."""
        trainer.labelled_calculations = []
        assert mlip.training_metrics() == dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)

    def test_training_metrics(self, mlip, trainer):
        """RMSE energy and forces are computed from the deployed potential's predictions against the labels."""
        trainer.labelled_calculations = [make_calculation(0.0, [[0, 0, 0]]),
                                         make_calculation(0.0, [[0, 0, 0]])]
        predictions = [make_calculation(1.0, [[1, 0, 0]]), make_calculation(3.0, [[0, 0, 0]])]

        with patch.object(mlip, "calculate", return_value=predictions) as calculate:
            metrics = mlip.training_metrics()

        calculate.assert_called_once()
        assert metrics["n_training_conf"] == 2
        assert metrics["rmse_energy"] == pytest.approx(np.sqrt((1.0 ** 2 + 3.0 ** 2) / 2))
        assert metrics["rmse_forces"] == pytest.approx(np.sqrt(1.0 ** 2 / 6))


class TestDerivedMLIP:

    @pytest.fixture(params=[pytest.param("flare", marks=pytest.mark.requires_flare)])
    def mlip_type(self, request):
        return request.param

    @pytest.fixture
    def trainer(self, mlip_type, structure, list_element_symbols):
        if mlip_type == "flare":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import (
                FlareConfiguration, FlareTrainer)
            trainer = FlareTrainer(FlareConfiguration(cutoff=5.0,
                                                      elements=list_element_symbols,
                                                      n_radial=8,
                                                      lmax=3,
                                                      variance_type='local'))
            labelled_structure = SinglePointCalculation(calculation_type="dummy_test",
                                                        structure=structure,
                                                        forces=np.random.rand(len(structure), 3),
                                                        energy=-1.0)
            trainer.add_labelled_structure(labelled_structure, active_environment_indices=list(range(len(structure))))
            trainer.fit()  # a pretrained model, ready to deploy
            return trainer
        raise ValueError(f"Unknown MLIP type '{mlip_type}'.")

    @pytest.fixture
    def mlip(self, mlip_type, trainer):
        if mlip_type == "flare":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
                FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_mlip import \
                FlareMLIP
            optimizer = FlareHyperparametersOptimizer(
                FlareOptimizerConfiguration(max_optimization_iterations=3,
                                            optimize_sigma=False,
                                            optimize_sigma_e=True,
                                            optimize_sigma_f=False,
                                            optimize_sigma_s=False)
            )
            return FlareMLIP(flare_trainer=trainer, hyperparameter_optimizer=optimizer, lammps_runner=MagicMock())
        raise ValueError(f"Unknown MLIP type '{mlip_type}'.")

    def test_prepare_mlip_first_round(self, mlip, tmp_path):
        """prepare_mlip_first_round deploys the pretrained model to a runnable potential, without training."""
        mlip.prepare_mlip_first_round(tmp_path)

        assert isinstance(mlip.lammps_potential, LammpsPotential)
        assert len(list(tmp_path.iterdir())) > 0

    def test_train(self, mlip_type, mlip, trainer, tmp_path):
        """train fits the model, deploys it and writes a checkpoint."""
        with ExitStack() as stack:
            fit = stack.enter_context(patch.object(trainer, "fit", wraps=trainer.fit))
            fit_hyperparameters = None
            if mlip_type == "flare":
                fit_hyperparameters = stack.enter_context(
                    patch.object(trainer, "fit_hyperparameters", wraps=trainer.fit_hyperparameters))
            mlip.train(tmp_path)

        fit.assert_called_once()
        if fit_hyperparameters is not None:
            fit_hyperparameters.assert_called_once()
        assert isinstance(mlip.lammps_potential, LammpsPotential)
        assert (tmp_path / "checkpoint.json").is_file()
