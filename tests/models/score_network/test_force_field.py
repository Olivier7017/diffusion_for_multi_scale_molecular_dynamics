import pytest
from pathlib import Path
import numpy as np
import time
from functools import wraps

from ase import Atoms
from ase.io import Trajectory, read, write
from ase.calculators.emt import EMT
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from pytest_mock import mocker
from unittest.mock import patch
import torch

from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.ase_for_diffusion_data_module import \
    ASEForDiffusionDataModuleParameters, ASEForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import EGNNScoreNetworkParameters, EGNNScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import ZBLForce, ZBLForceParameters
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import AtomTypeLossParameters, MSELossParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import \
    ReduceLROnPlateauSchedulerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import AXLDiffusionParameters
from diffusion_for_multi_scale_molecular_dynamics.callbacks.callback_loader import create_all_callbacks
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.sample_maker_factory import create_sample_maker, create_sample_maker_parameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.nearest_neighbors_excisor import NearestNeighborsExcisionArguments
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.top_k_atom_selector import TopKAtomSelectorParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.data.utils import traj_to_AXL, AXL_to_traj
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_lattice_parameters_to_unit_cell_vectors)


class TestForceFieldAugmentedScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture()
    def atomic_positions(self):
        pos = [[[2.0722, 6.8944, 4.1256],
               [3.5923, 1.7706, 3.1422],
               [5.6965, 5.8560, 1.6302],
               [1.1049, 5.1537, 6.9432],
               [5.3291, 0.0841, 0.0237],
               [6.9115, 6.2579, 6.8071],
               [0.9318, 3.6683, 3.3170],
               [4.9636, 2.1880, 6.4710],
               [5.9502, 0.8173, 0.6279],
               [5.7841, 2.8497, 3.4793],
               [4.6890, 1.1964, 1.4846],
               [2.4854, 1.0776, 4.6584],
               [2.7329, 4.4024, 6.6043],
               [2.2355, 6.9130, 2.1594],
               [0.7413, 5.7150, 2.4136],
               [0.9003, 2.8371, 3.7896],
               [4.9723, 1.1048, 6.9392],
               [2.1095, 1.8250, 0.0095],
               [1.2997, 1.5873, 2.0805],
               [3.6191, 4.1745, 2.1670],
               [5.9250, 1.3836, 3.8350],
               [1.6283, 2.3922, 5.5159],
               [2.0133, 0.2571, 6.3284],
               [0.0960, 2.0796, 3.2575],
               [3.5847, 3.7682, 3.8876],
               [6.1058, 5.0023, 5.3827],
               [1.8365, 2.1038, 3.7810],
               [0.3789, 2.4084, 4.3938],
               [6.4045, 6.2634, 0.3362],
               [3.1300, 4.4086, 0.7728],
               [2.8324, 1.7593, 6.0379],
               [3.5303, 4.5427, 4.7823]]]
        return pos

    @pytest.fixture()
    def number_of_samples(self, atomic_positions):
        return len(atomic_positions)

    @pytest.fixture()
    def number_of_atoms(self, atomic_positions):
        return len(atomic_positions[0])

    @pytest.fixture()
    def basis_vectors(self):
        vec =[[[7.2200, 0.0000, 0.0000],
             [0.0000, 7.2200, 0.0000],
             [0.0000, 0.0000, 7.2200]]]
        return torch.tensor(vec, dtype=torch.float32, device="cpu")

    @pytest.fixture()
    def element_list(self):
        return ["Si"]

    @pytest.fixture()
    def number_of_elements(self, element_list):
        return len(element_list)

    @pytest.fixture()
    def times(self):
        return [1e-5, 0.5, 1.]

    @pytest.fixture()
    def atoms(self, element_list, number_of_atoms, atomic_positions, basis_vectors):
        atoms = Atoms(
            symbols=element_list*number_of_atoms,
            positions=atomic_positions[0],
            cell=basis_vectors[0],
            pbc=True
        )
        return atoms

    @pytest.fixture()
    def lattice_parameters(self, basis_vectors):
        lattice_parameters = torch.tensor([[basis_vectors[0, 0, 0], basis_vectors[0, 1, 1],
                                            basis_vectors[0, 2, 2], basis_vectors[0, 1, 2],
                                            basis_vectors[0, 0, 2], basis_vectors[0, 0, 1]]])
        return lattice_parameters

    @pytest.fixture()
    def reduced_positions(self, atoms, basis_vectors):
        return (torch.from_numpy(atoms.get_scaled_positions(wrap=True))[None, ...]).to(basis_vectors)

    @pytest.fixture()
    def noise_parameters(self):
        noise_parameters = NoiseParameters(
            total_time_steps = 3,
            sigma_min = 0.005,
            sigma_max = 0.5,
            schedule_type = "exponential",
        )
        return noise_parameters

    @pytest.fixture()
    def score_network_parameters(self):
        score_network_parameters = EGNNScoreNetworkParameters(
                                       num_atom_types=1,
                                       number_of_bloch_wave_shells=1,
                                       message_n_hidden_dimensions=2,
                                       message_hidden_dimensions_size=64,
                                       node_n_hidden_dimensions=2,
                                       node_hidden_dimensions_size=64,
                                       coordinate_n_hidden_dimensions=2,
                                       coordinate_hidden_dimensions_size=64,
                                       residual=True,
                                       attention=False,
                                       normalize=True,
                                       tanh=True,
                                       coords_agg="mean",
                                       message_agg="mean",
                                       n_layers=4,
                                       edges="radial_cutoff",
                                       radial_cutoff=5.,
                                       drop_duplicate_edges=True)
        return score_network_parameters

    @pytest.fixture()
    def sampling_parameters(self, number_of_samples, number_of_atoms, basis_vectors):
        sampling_parameters = PredictorCorrectorSamplingParameters(number_of_samples=number_of_samples,
                                                                   spatial_dimension=3,
                                                                   number_of_corrector_steps=1,
                                                                   num_atom_types=1,
                                                                   number_of_atoms=number_of_atoms,
                                                                   use_fixed_lattice_parameters=True,
                                                                   cell_dimensions=basis_vectors[0],
                                                                   record_samples=True)
        return sampling_parameters

    @pytest.fixture()
    def composition(self, number_of_atoms, reduced_positions, lattice_parameters):
        A = torch.tensor([[1]*number_of_atoms])
        return AXL(A=A, X=reduced_positions, L=lattice_parameters)

    def test_Si_forcefield_ZBL(self, number_of_samples, number_of_atoms, number_of_elements,
                               basis_vectors, element_list, times,
                               score_network_parameters, sampling_parameters, noise_parameters,
                               composition):
        """Test for ZBLRepulsionScore using masked_atom type of 14.5"""
        # 1. Prepare the objects for the test
        noise_sched = NoiseScheduler(noise_parameters, num_classes=number_of_elements+1)
        noise, _ = noise_sched.get_all_sampling_parameters()
        sigmas = noise.sigma
        forces = torch.zeros([number_of_atoms, 3]).to(basis_vectors)
    
        zbl_parameters = ZBLForceParameters(
            cutoff_radius=2.19293,
            inner_radius_fraction=0.5552844824048191,
            element_list=element_list,
            device="cpu",
        )
        zbl_force = ZBLForce(zbl_parameters)

        score_network = EGNNScoreNetwork(score_network_parameters)
        model = ForceFieldAugmentedScoreNetwork(
            score_network=score_network,
            score_forces=zbl_force,
            diffusion_time_scaling="linear",
            force_activation_scale=100.,
            use_for_training=False,
        )
    
        model.eval()  # Set model.training to False
        fake_model_output = AXL(
            A=torch.zeros([number_of_samples, number_of_atoms, number_of_elements]).to(basis_vectors),
            X=torch.zeros([number_of_samples, number_of_atoms, 3]).to(basis_vectors),
            L=torch.zeros([1, 6]).to(basis_vectors)
        )  # We want the EGNN model to return a zeros
    
        generator = LangevinGenerator(noise_parameters=noise_parameters,
                                      sampling_parameters=sampling_parameters,
                                      axl_network=model)
    
        # 2. Calculate the scores and make an updated structure
        force_scores, updated_structs = [], []
        g2_squared_test = [1e-4, 1e-5, 1e-6]  # The g2_i decreases with t->0. Override to get predictability
        with patch.object(model._score_network, "forward", return_value=fake_model_output):
            for i in range(len(times)):
                time_tensor = (times[i] * torch.ones(number_of_samples, 1)).to(composition.X)
                sigma_tensor = sigmas[i] * torch.ones_like(time_tensor)
                batch = {NOISY_AXL_COMPOSITION: composition,
                         TIME: time_tensor,
                         NOISE: sigma_tensor,
                         CARTESIAN_FORCES: forces,
                }
    
                force_score = model(batch)
                force_scores.append(force_score)
    
                sigma_i = noise_sched._sigma_array[i]
    
                g2i_updated_structs = []
                for g2_i in g2_squared_test:
                    score_weight = g2_i *torch.ones_like(noise_sched._g_squared_array[i])
                    gaussian_noise = torch.zeros_like(noise_sched._g_array[i])  # We don't want to introduce the gaussian_noise
                    z_noise = torch.zeros_like(force_score.X)
    
                    g2i_updated_structs.append(
                        generator._relative_coordinates_update(
                            relative_coordinates=composition.X,
                            sigma_normalized_scores=force_score.X,
                            sigma_i =sigma_i,
                            score_weight=score_weight,
                            gaussian_noise_weight=gaussian_noise,
                            z=z_noise,
                        )
                    )
                updated_structs.append(g2i_updated_structs)
    
        # 3. Verify the results
        # Note : We do small displacements and assume forces should decrease with bigger g2_i or smaller t.
    
        # 3.1 Verify the maximal force decreases more with bigger g2_i
        for g2i_updated_structs in updated_structs:
            force_g2i_small = zbl_force.get_forces(composition.A, g2i_updated_structs[0], basis_vectors)
            force_g2i_smaller = zbl_force.get_forces(composition.A, g2i_updated_structs[1], basis_vectors)
            force_g2i_smallest = zbl_force.get_forces(composition.A, g2i_updated_structs[2], basis_vectors)
    
            assert force_g2i_small.abs().max() <= force_g2i_smaller.abs().max()
            assert force_g2i_smaller.abs().max() <= force_g2i_smallest.abs().max()
    
        # 3.2 Verify that the maximal force decreases more with t->0 (only true because every g2_i is equal)
        force_zerotime = zbl_force.get_forces(composition.A, updated_structs[0][0], basis_vectors)
        force_halftime = zbl_force.get_forces(composition.A, updated_structs[1][0], basis_vectors)
        force_Ttime = zbl_force.get_forces(composition.A, updated_structs[2][0], basis_vectors)
    
        assert force_zerotime.abs().max() <= force_halftime.abs().max()
        assert force_halftime.abs().max() <= force_Ttime.abs().max()
        
        # 3.3 Verify that ZBL force didn't change the struct at t=T (gaussian noise was also removed for this test)
        Ttime_struct = updated_structs[2][0]
        assert torch.allclose(Ttime_struct, composition.X, atol=1e-4)
    
        # 3.4 Verify that the minimal interatomic distances is bigger in the updated_struct
        initial_dist = zbl_force.get_atomic_distances(composition.X, basis_vectors)
        zerotime_dist = zbl_force.get_atomic_distances(updated_structs[0][0], basis_vectors)
        halftime_dist = zbl_force.get_atomic_distances(updated_structs[1][0], basis_vectors)
    
        # Here, we need to filter out atoms outside rcut (dist=-1) with a mask
        initial_min_dist = initial_dist.masked_fill(initial_dist < 0, float("inf")).min()
        zerotime_min_dist = zerotime_dist.masked_fill(zerotime_dist < 0, float("inf")).min()
        halftime_min_dist = halftime_dist.masked_fill(halftime_dist < 0, float("inf")).min()
    
        assert zerotime_min_dist >= halftime_min_dist
        assert halftime_min_dist >= initial_min_dist


    def test_ZBL_with_no_forces(self, number_of_samples, number_of_atoms, number_of_elements,
                           basis_vectors, element_list, score_network_parameters, sampling_parameters, noise_parameters,
                           composition):
        """Smoke test for ForceFieldAugmentedScoreNetwork + ZBL with no interacting atoms."""
        # 1. Create the object
        time = 1e-4
        sigma = 5e-3
        g2_i = 1e-4
        noise_sched = NoiseScheduler(noise_parameters, num_classes=number_of_elements+1)
        noise, _ = noise_sched.get_all_sampling_parameters()
        forces = torch.zeros([number_of_atoms, 3]).to(basis_vectors)


        zbl_parameters = ZBLForceParameters(
            cutoff_radius=1e-4,  # Tiny cutoff_radius so there's no interacting pairs
            inner_radius_fraction=0.5,
            element_list=element_list,
            device="cpu",
        )
        zbl_force = ZBLForce(zbl_parameters)

        score_network = EGNNScoreNetwork(score_network_parameters)
        model = ForceFieldAugmentedScoreNetwork(
            score_network=score_network,
            score_forces=zbl_force,
            diffusion_time_scaling="linear",
            force_activation_scale=100.0,
            use_for_training=False,
        )
        model.eval()

        generator = LangevinGenerator(noise_parameters=noise_parameters,
                                      sampling_parameters=sampling_parameters,
                                      axl_network=model)

        # Patch EGNN to output zeros so any non-zero would have to come from ZBL
        fake_model_output = AXL(
            A=torch.zeros(
                (number_of_samples, number_of_atoms, number_of_elements),
                device=basis_vectors.device,
                dtype=basis_vectors.dtype,
            ),
            X=torch.zeros(
                (number_of_samples, number_of_atoms, 3),
                device=basis_vectors.device,
                dtype=basis_vectors.dtype,
            ),
            L=torch.zeros_like(composition.L),
        )

        # 2. Do the calculations
        with patch.object(model._score_network, "forward", return_value=fake_model_output):
            time_tensor = (time * torch.ones(number_of_samples, 1)).to(composition.X)
            sigma_tensor = sigma * torch.ones_like(time_tensor)
            batch = {NOISY_AXL_COMPOSITION: composition,
                     TIME: time_tensor,
                     NOISE: sigma_tensor,
                     CARTESIAN_FORCES: forces,
            }

            force_score = model(batch)


            score_weight = g2_i * torch.ones_like(noise_sched._g_squared_array)
            gaussian_noise = torch.zeros_like(noise_sched._g_array)  # We don't want to introduce the gaussian_noise
            z_noise = torch.zeros_like(force_score.X)

            updated_struct = generator._relative_coordinates_update(
                    relative_coordinates=composition.X,
                    sigma_normalized_scores=force_score.X,
                    sigma_i =sigma,
                    score_weight=score_weight,
                    gaussian_noise_weight=gaussian_noise,
                    z=z_noise,
                )

        # 3. Assert everything works as expected
        # 3.1 The force_score should filled with 0.
        assert torch.allclose(force_score.X, torch.zeros_like(force_score.X), atol=1e-4)

        # 3.2 The updated_struct should be identical to the initial one
        assert torch.allclose(updated_struct, composition.X, atol=1e-4)
