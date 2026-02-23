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

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.ase_for_diffusion_data_module import \
    ASEForDiffusionDataModuleParameters, ASEForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import EGNNScoreNetworkParameters, EGNNScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import ZBLForce
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


def main():
    #Si4()
    #Si()
    #SiGe4()
    #SiGe()
    Si_forcefield_32_score()


def masked_type_32():
    repulsion_score = ZBLForce(
        cutoff_radius=2.19293,
        inner_radius_fraction=0.5552844824048191,
        element_list=["Si", "P"],
        device="cpu",
    )
    elem = [3, 3, 1, 3, 1, 3, 1, 1, 2, 1, 2, 1, 2, 3, 3, 2, 3, 1, 2, 1, 3, 2, 3, 1, 1, 2, 3, 2, 2, 2, 3, 1]
    A = torch.tensor([elem], dtype=torch.long)
    pos, vec = get_pos_and_basis_vectors()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)

    torch_forces = repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0].detach().cpu().numpy()


def Si_forcefield_32_score():
    """Test for ZBLRepulsionScore using masked_atom type of 14.5"""
    number_of_samples = 1
    element_list=["Si"]
    times = [1e-5, 0.5, 1.]
    natoms = 32
    nelems = len(element_list)

    # 1. Prepare everything for the batch object (we just need to wait for the loop on times to actually create it
    A = torch.tensor([[1]*natoms])
    pos, vec = get_pos_and_basis_vectors()
    basis_vectors = torch.tensor(vec)
    atoms = Atoms(
        symbols=element_list*natoms,
        positions=pos[0],
        cell=vec[0],
        pbc=True
    )
    reduced_positions = torch.tensor([atoms.get_scaled_positions(wrap=True)]).to(basis_vectors)

    lattice_params = torch.tensor([[basis_vectors[0, 0, 0], basis_vectors[0, 1, 1], basis_vectors[0, 2, 2],
                                   basis_vectors[0, 1, 2], basis_vectors[0, 0, 2], basis_vectors[0, 0, 1]]])
    composition = AXL(A=A, X=reduced_positions, L=lattice_params)

    noise_params = NoiseParameters(
        total_time_steps = 3,
        sigma_min = 0.005,
        sigma_max = 0.5,
        schedule_type = "exponential",
    )
    noise_sched = NoiseScheduler(noise_params, num_classes=nelems+1)
    noise_sched = NoiseScheduler(noise_params, num_classes=nelems+1)
    noise, _ = noise_sched.get_all_sampling_parameters()
    sigmas = noise.sigma
    forces = torch.zeros_like(reduced_positions)

    # 2. Create the model
    zbl_force = ZBLForce(
        cutoff_radius=2.19293,
        inner_radius_fraction=0.5552844824048191,
        element_list=element_list,
        device="cpu",
    )
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
    score_network = EGNNScoreNetwork(score_network_parameters)
    model = ForceFieldAugmentedScoreNetwork(
        score_network=score_network,
        score_forces=zbl_force,
        diffusion_time_scaling="linear",
        force_activation_scale=100.,
        use_for_training=False,
    )

    model.eval()  # Set model.training to False
    fake_output = AXL(A=torch.zeros([number_of_samples, len(pos), nelems]), 
                      X=torch.zeros_like(reduced_positions),
                      L=torch.zeros([1, 6])
    )  # We want the EGNN model to return a zeros

    # 3. Create a LangevinGenerator
    sampling_params = PredictorCorrectorSamplingParameters(number_of_samples=number_of_samples,
                                                           spatial_dimension=3,
                                                           number_of_corrector_steps=1,
                                                           num_atom_types=1,
                                                           number_of_atoms=natoms,
                                                           use_fixed_lattice_parameters=True,
                                                           cell_dimensions=vec[0],
                                                           record_samples=True)
    generator = LangevinGenerator(noise_parameters=noise_params,
                                  sampling_parameters=sampling_params,
                                  axl_network=model)

    # 4. Calculate the scores and look the updated structure
    force_scores, updated_structs = [], []
    g2_squared_test = [1e-4, 1e-5, 1e-6]  # The g2_i decreases with t->0. I'll remove this for testing
    with patch.object(model._score_network, "forward", return_value=fake_output):
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
                score_weight = g2_squared_test[1] *torch.ones_like(noise_sched._g_squared_array[i])
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

    # 5. Verify the results
    # Note : We do small displacements and assume forces should decrease with bigger g2_i or smaller t.

    # 5.1 Verify the maximal force decreases more with bigger g2_i
    for g2i_updated_structs in updated_structs:
        force_g2i_small = zbl_force.get_forces(composition.A, g2i_updated_structs[0], basis_vectors)
        force_g2i_smaller = zbl_force.get_forces(composition.A, g2i_updated_structs[1], basis_vectors)
        force_g2i_smallest = zbl_force.get_forces(composition.A, g2i_updated_structs[2], basis_vectors)

        assert force_g2i_small.abs().max() <= force_g2i_smaller.abs().max()
        assert force_g2i_smaller.abs().max() <= force_g2i_smallest.abs().max()

    # 5.2 Verify that the maximal force decreases more with t->0 (only true because every g2_i is equal)
    force_zerotime = zbl_force.get_forces(composition.A, updated_structs[0][0], basis_vectors)
    force_halftime = zbl_force.get_forces(composition.A, updated_structs[1][0], basis_vectors)
    force_Ttime = zbl_force.get_forces(composition.A, updated_structs[2][0], basis_vectors)

    assert force_zerotime.abs().max() <= force_halftime.abs().max()
    assert force_halftime.abs().max() <= force_Ttime.abs().max()
    
    # 5.3 Verify that ZBL force didn't change the struct at t=T (gaussian noise was also removed for this test)
    Ttime_struct = updated_structs[2][0]
    assert torch.allclose(Ttime_struct, composition.X, atol=1e-4)

    # 5.4 Verify that the minimal interatomic distances is bigger in the updated_struct
    initial_dist = zbl_force.get_atomic_distances(composition.X, basis_vectors)
    zerotime_dist = zbl_force.get_atomic_distances(updated_structs[0][0], basis_vectors)
    halftime_dist = zbl_force.get_atomic_distances(updated_structs[1][0], basis_vectors)

    # Here, we need to filter out atoms outside rcut (dist=-1) with a mask
    initial_min_dist = initial_dist.masked_fill(initial_dist < 0, float("inf")).min()
    zerotime_min_dist = zerotime_dist.masked_fill(zerotime_dist < 0, float("inf")).min()
    halftime_min_dist = halftime_dist.masked_fill(halftime_dist < 0, float("inf")).min()

    assert zerotime_min_dist >= halftime_min_dist
    assert halftime_min_dist >= initial_min_dist

    # Verify that Zbl_force=0 doesn't create a crash

    print("TEST COMPLETED")   

def Si4():
    repulsion_score=ZBLRepulsionScore(cutoff_radius=2.19293, inner_radius_fraction=0.5552844824048191, element_list=["Si"])
    A = torch.tensor([[1]*4])
    pos, vec = get_si4_pos_and_basis_vectors()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)
    dfmsmd_forces = np.array(repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0])
    lammps_forces = read_lammps_forces("/home/olivi/projects/00-Debug/04-TestZBL/Si4.dump")

    assert np.allclose(dfmsmd_forces, lammps_forces, rtol=1e-2, atol=1e-6)
    [print(f"{dfmsmd_forces[i]}, {lammps_forces[i]}") for i in range(len(dfmsmd_forces))]

def Si():
    repulsion_score=ZBLRepulsionScore(cutoff_radius=2.19293, inner_radius_fraction=0.5552844824048191, element_list=["Si"])
    A = torch.tensor([[1]*32])
    pos, vec = get_pos_and_basis_vectors()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)
    dfmsmd_forces = np.array(repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0])
    lammps_forces = read_lammps_forces("/home/olivi/projects/00-Debug/04-TestZBL/Si.dump")

    assert np.allclose(dfmsmd_forces, lammps_forces, rtol=1e-2, atol=1e-6)
    [print(f"{dfmsmd_forces[i]}, {lammps_forces[i]}") for i in range(len(dfmsmd_forces))]


def SiGe():
    repulsion_score=ZBLRepulsionScore(cutoff_radius=2.19293, inner_radius_fraction=0.5552844824048191, element_list=["Si", "Ge"])
    A = torch.tensor([[2,1]*16])
    pos, vec = get_pos_and_basis_vectors()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)
    #print(cartesian_positions)
    #print(basis_vectors)
    #print(A)
    dfmsmd_forces = np.array(repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0])
    lammps_forces = read_lammps_forces("/home/olivi/projects/00-Debug/04-TestZBL/SiGe.dump")

    #assert np.allclose(dfmsmd_forces, lammps_forces, rtol=1e-2, atol=1e-6)
    [print(f"{dfmsmd_forces[i]}, {lammps_forces[i]}") for i in range(len(dfmsmd_forces))]


def SiGe4():
    repulsion_score=ZBLRepulsionScore(cutoff_radius=2.19293, inner_radius_fraction=0.5552844824048191, element_list=["Si", "Ge"])
    #A = torch.tensor([[1,2]*2])
    A = torch.tensor([[1,1,1,2]])
    pos, vec = get_si4_pos_and_basis_vectors()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)
    #print(cartesian_positions)
    #print(basis_vectors)
    #print(A)
    dfmsmd_forces = np.array(repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0])
    lammps_forces = read_lammps_forces("/home/olivi/projects/00-Debug/04-TestZBL/SiGe4.dump")

    #assert np.allclose(dfmsmd_forces, lammps_forces, rtol=1e-2, atol=1e-6)
    [print(f"{dfmsmd_forces[i]}, {lammps_forces[i]}") for i in range(len(dfmsmd_forces))]


def read_lammps_forces(fn):
    with open(fn, "r") as f:
        lines = f.readlines()

    forces = []
    start_read = False
    for line in lines:
        if line.strip().startswith("ITEM: ATOMS"):
            start_read=True
            continue
        if start_read:
            forces.append([float(x) for x in line.split()[-3:]])
    return np.array(forces)

def get_pos_and_basis_vectors():
    """I want semi-random pos and basis vectors, but without recomputing the lammps part"""
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
    vec =[[[7.2200, 0.0000, 0.0000],
         [0.0000, 7.2200, 0.0000],
         [0.0000, 0.0000, 7.2200]]]
    return pos, vec 

def get_si4_pos_and_basis_vectors():
    pos = [[[2.0722, 6.8944, 4.1256],
           [6.9115, 6.2579, 6.8071],
           [2.4854, 1.0776, 4.6584],
           [6.4045, 6.2634, 0.3362]]]
    vec =[[[7.2200, 0.0000, 0.0000],
         [0.0000, 7.2200, 0.0000],
         [0.0000, 0.0000, 7.2200]]]
    return pos, vec

if __name__=="__main__":
    main()
