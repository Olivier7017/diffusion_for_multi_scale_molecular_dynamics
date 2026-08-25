"""Active learning example: FLARE MLIP, MD dynamic driver, Stillinger-Weber oracle, no-op sample maker.

This example is self-contained: it creates its own starting configuration and starts the MLIP cold. Set the
values in the "User configuration" block below for your machine, then run this file. Every create_* function
lists all of its parameters (defaults left explicit) so the available knobs are visible; main() is identical
across the examples, so switching a component is a copy-paste of the matching create_* function.

Notes about this example (the chosen options):
    - MLIP: FLARE, started cold (a fresh model). Option B in create_mlip loads a pretrained checkpoint instead.
    - Dynamic driver: MD (NVT molecular dynamics).
    - Oracle: Stillinger-Weber.
    - Sample maker: no-op, which keeps the uncertain structure as-is (no excision or repaint).
"""

from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.md_driver.md_driver import \
    MdDriver
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_configuration import \
    FlareConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_mlip import \
    FlareMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
    FlareTrainer
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)

# --- User configuration (set these for your machine and task) ---
ELEMENT_LIST = ["Si"]
UNCERTAINTY_THRESHOLD = 0.1
WORKING_DIRECTORY = Path("active_learning_run")
REFERENCE_DIRECTORY = Path("reference")  # where the generated initial_configuration.dat is written
LAMMPS_EXECUTABLE_PATH = Path("/path/to/lmp")  # your LAMMPS executable
STILLINGER_WEBER_COEFFICIENTS_FILE_PATH = Path("Si.sw")  # your Stillinger-Weber coefficients file


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
    write_initial_configuration(create_atoms())
    active_learning = ActiveLearning(
        oracle_single_point_calculator=create_oracle(),
        sample_maker=create_sample_maker(),
        dynamic_driver=create_dynamic_driver(),
    )
    active_learning.run_campaign(
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        mlip=create_mlip(),
        working_directory=WORKING_DIRECTORY,
        maximum_number_of_rounds=100,
        restart_from_stage="auto",  # either {"auto", "driver", "oracle", "train"}
    )


def create_atoms():
    """Create the starting atomic structure (a small silicon crystal)."""
    structure = Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    structure.make_supercell([2, 2, 2])
    return structure


def write_initial_configuration(structure):
    """Write the structure as REFERENCE_DIRECTORY/initial_configuration.dat (the driver's starting point)."""
    REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
    lammps_data.write_file(str(REFERENCE_DIRECTORY / "initial_configuration.dat"))


def create_mlip():
    """Create the FLARE MLIP to drive and refine."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )
    hyperparameter_optimizer = FlareHyperparametersOptimizer(FlareOptimizerConfiguration(
        optimization_method="BFGS", max_optimization_iterations=100,
        optimize_sigma=True, optimize_sigma_e=True, optimize_sigma_f=True, optimize_sigma_s=True,
        print=False, ftol=1e-3, gtol=1e-3,
    ))

    # A. Cold start from a fresh FLARE model (default).
    flare_configuration = FlareConfiguration(
        cutoff=5.0, elements=ELEMENT_LIST, n_radial=8, lmax=3, variance_type="local",
        initial_sigma=1.0, initial_sigma_e=1e-2, initial_sigma_f=1e-3, initial_sigma_s=1e-1,
    )
    flare_trainer = FlareTrainer(flare_configuration=flare_configuration)
    return FlareMLIP(flare_trainer=flare_trainer, hyperparameter_optimizer=hyperparameter_optimizer,
                     lammps_runner=lammps_runner)

    # B. Or load a pretrained FLARE checkpoint instead of the cold start above:
    # return FlareMLIP.load_checkpoint(
    #     Path("initial_flare/checkpoint.json"),
    #     hyperparameter_optimizer=hyperparameter_optimizer,
    #     lammps_runner=lammps_runner,
    # )


def create_dynamic_driver():
    """Create a molecular-dynamics dynamic driver."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )
    return MdDriver(lammps_runner=lammps_runner, reference_directory=REFERENCE_DIRECTORY,
                    temperature=300.0, timestep=0.001, number_of_steps=1000)


def create_oracle():
    """Create a Stillinger-Weber oracle."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1, mpi_executable="mpirun",
    )
    # Or run LAMMPS in-process (lower overhead); requires the LAMMPS python binding ('import lammps'):
    # from diffusion_for_multi_scale_molecular_dynamics.calc.lammps_runner import InProcessLammpsRunner
    # lammps_runner = InProcessLammpsRunner()
    potential = StillingerWeberPotential(sw_coefficients_file_path=STILLINGER_WEBER_COEFFICIENTS_FILE_PATH)
    return LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=lammps_runner,
                                       with_uncertainty=False)


def create_sample_maker():
    """Create a no-op sample maker (labels the uncertain structure itself, without excision or repaint)."""
    atom_selector = ThresholdAtomSelector(ThresholdAtomSelectorParameters(
        algorithm="threshold", uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    ))
    arguments = NoOpSampleMakerArguments(
        element_list=ELEMENT_LIST, algorithm="noop", sample_box_strategy="noop", sample_box_size=None,
    )
    return NoOpSampleMaker(sample_maker_arguments=arguments, atom_selector=atom_selector)


if __name__ == "__main__":
    main()
