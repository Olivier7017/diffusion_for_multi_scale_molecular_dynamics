"""Active learning example: MTP MLIP, ARTn dynamic driver, Stillinger-Weber oracle, excise-and-random sampler.

This example is self-contained and every create_* function is interchangeable with other activelearning_example
lists all of its parameters (defaults left explicit) so the available knobs are visible; main() is identical
across the examples, so switching a component is a copy-paste of the matching create_* function.

Required packages (beyond the base install):
    - MLIP-3, providing the 'mlp' executable that fits the MTP: https://gitlab.com/ashapeev/mlip-3.git
    - lammps-mtp-kokkos, a LAMMPS with the mtp/extrapolation pair_style (can be built CPU-only):
      https://github.com/RichardZJM/lammps-mtp-kokkos.git
    - artn-plugin, the compiled ARTn LAMMPS plugin: https://gitlab.com/mammasmias/artn-plugin.git

Notes about this example (the chosen options):
    - MLIP: MTP, started cold (a fresh model). Option B in create_mlip loads a pretrained potential instead.
    - Dynamic driver: ARTn (activation-relaxation saddle search).
    - Oracle: Stillinger-Weber.
    - Sample maker: excise-and-random, which cuts out the uncertain environments and refills a box at random.
"""

from pathlib import Path

from ase.build import bulk
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.artn_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_mlip import \
    MtpMlip
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
    MtpTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.excise_and_random_sample_maker import (
    ExciseAndRandomSampleMaker, ExciseAndRandomSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.excisor.nearest_neighbors_excisor import (
    NearestNeighborsExcision, NearestNeighborsExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_pymatgen_structure
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    label_configurations

# --- User configuration (set these for your machine and task) ---
ELEMENT_LIST = ["Si"]
UNCERTAINTY_THRESHOLD = 2.0  # MTP extrapolation grade (gamma); atoms above this are treated as uncertain
WORKING_DIRECTORY = Path("active_learning_run")
REFERENCE_DIRECTORY = Path("reference")  # where the generated initial_configuration.dat is written
LAMMPS_EXECUTABLE_PATH = Path("/path/to/lmp")  # your LAMMPS executable (built with mtp/extrapolation and ARTn)
STILLINGER_WEBER_COEFFICIENTS_FILE_PATH = Path("Si.sw")  # your Stillinger-Weber coefficients file
MLP_EXECUTABLE_PATH = Path("/path/to/mlp")  # the MLIP-3 'mlp' executable (fits the MTP)
ARTN_LIBRARY_PLUGIN_PATH = Path("/path/to/libartn-lmp.so")  # compiled ARTn plugin (or set ARTN_PLUGIN_PATH)


def main():
    """Run one active learning campaign (identical in every example).

    The campaign repeats these steps until the dynamic driver finds no uncertain structure:
        Stage.DRIVER (run_dynamic_driver):
            1. Run the dynamic driver with the MLIP.
            2. Extract the uncertainty per atom.
        Stage.ORACLE (oracle_evaluation):
            3. Make samples from the uncertain structure.
            4. Label the samples with the oracle.
            5. Commit the labelled structures to the training database.
        Stage.TRAIN (_retrain):
            6. Fold the labelled structures into the model and retrain the MLIP.

    With restart_from_stage="auto" the campaign starts from Stage.DRIVER on a fresh run, and on a restart
    resumes just after the last stage that completed successfully.
    """
    initial_configuration = create_initial_configuration()  # DynamicDriver initial configuration
    provided_configurations = create_provided_configurations()  # Initial training configuration for the MLIP

    oracle = create_oracle()
    provided_configurations = label_configurations(create_atoms(), oracle)
    exit()

    active_learning = ActiveLearning(
        oracle_single_point_calculator=oracle,
        sample_maker=create_sample_maker(),
        dynamic_driver=create_dynamic_driver(),
    )
    active_learning.run_campaign(
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        mlip=create_mlip(),
        working_directory=WORKING_DIRECTORY,
        provided_configurations=provided_configurations,
        maximum_number_of_rounds=100,
        restart_from_stage="auto",  # either {"auto", "driver", "oracle", "train"}
    )


def create_initial_configuration():
    """Create the starting configuration(s) as an ase trajectory (a small silicon crystal)."""
    silicon = bulk("Si", "diamond", a=5.43)
    silicon_supercell = silicon.repeat((2, 2, 2))
    return silicon_supercell

def create_provided_configurations():
    """Create the starting configuration(s) as an ase trajectory (a small silicon crystal)."""
    silicon = bulk("Si", "diamond", a=5.43)
    silicon_supercell = silicon.repeat((2, 2, 2))
    print(len(silicon_supercell))
    return [silicon_supercell]

def create_mlip():
    """Create the MTP MLIP to drive and refine."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )

    # A. Cold start from a fresh MTP model (default).
    mtp_configuration = MtpConfiguration(
        elements=ELEMENT_LIST, level=6, max_dist=5.0, min_dist=1.0,
        energy_weight=1.0, force_weight=0.01, stress_weight=0.0, site_en_weight=0.0,
        training_params=dict(max_iter=1000, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3),
    )
    mtp_trainer = MtpTrainer(mtp_configuration=mtp_configuration, mlp_executable_path=MLP_EXECUTABLE_PATH)
    return MtpMlip(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)

    # B. Or load a pretrained MTP potential instead of the cold start above (the configuration must be given,
    #    since the .almtp file does not record the training parameters):
    # mtp_trainer = MtpTrainer.load_checkpoint(
    #     Path("initial_mtp/potential.almtp"),
    #     mtp_configuration=MtpConfiguration(elements=ELEMENT_LIST, level=6, max_dist=5.0),
    #     mlp_executable_path=MLP_EXECUTABLE_PATH,
    # )
    # return MtpMlip(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)


def create_dynamic_driver():
    """Create an ARTn (activation-relaxation) dynamic driver."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )
    return ArtnDriver(
        lammps_runner=lammps_runner,
        reference_directory=REFERENCE_DIRECTORY,
        push_ids=0,  # index of the atom ARTn pushes to leave the initial basin
        push_add_const=[0.0, 0.0, 0.0, 1.0],  # four-component push constraint for that atom
        artn_library_plugin_path=ARTN_LIBRARY_PLUGIN_PATH,  # or None to read the ARTN_PLUGIN_PATH env var
        # ARTn namelist parameters, forwarded to the generated artn.in (see the ARTn documentation):
        engine_units="lammps/metal", verbose=2, ninit=2, lpush_final=True, nsmooth=2,
        forc_thr=0.01, push_step_size=0.1, push_mode="list", lanczos_disp=1e-4,
        lanczos_max_size=10, lanczos_min_size=3, lanczos_eval_conv_thr=1e-2,
        eigen_step_size=0.1, push_over=2.0,
    )


def create_oracle():
    """Create a Stillinger-Weber oracle."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )
    # Or run LAMMPS in-process (lower overhead); requires the LAMMPS python binding ('import lammps'):
    # from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import InProcessLammpsRunner
    # lammps_runner = InProcessLammpsRunner()
    potential = StillingerWeberPotential(sw_coefficients_file_path=STILLINGER_WEBER_COEFFICIENTS_FILE_PATH)
    return LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=lammps_runner,
                                       with_uncertainty=False)


def create_sample_maker():
    """Create an excise-and-random sample maker (cuts out uncertain environments, refills a box at random)."""
    atom_selector = TopKAtomSelector(TopKAtomSelectorParameters(
        algorithm="top_k", top_k_environment=4,  # keep the 4 most uncertain atoms as excision centers
    ))
    environment_excisor = NearestNeighborsExcision(NearestNeighborsExcisionArguments(
        algorithm="nearest_neighbors", number_of_neighbors=4,  # atoms kept around each excision center
    ))
    arguments = ExciseAndRandomSampleMakerArguments(
        element_list=ELEMENT_LIST, algorithm="excise_and_random",
        sample_box_strategy="fixed", sample_box_size=[6.0, 6.0, 6.0],  # cubic box (Angstrom) for each sample
        max_constrained_substructure=-1,  # -1: no limit on the number of excised environments
        number_of_samples_per_substructure=1,
        total_number_of_atoms=16,  # total atoms per sample (excised environment + random filler atoms)
        random_coordinates_algorithm="true_random",  # or "voxel_random"
        max_attempts=10, minimal_interatomic_distance=1.5,
    )
    return ExciseAndRandomSampleMaker(sample_maker_arguments=arguments, atom_selector=atom_selector,
                                      environment_excisor=environment_excisor)


if __name__ == "__main__":
    main()
