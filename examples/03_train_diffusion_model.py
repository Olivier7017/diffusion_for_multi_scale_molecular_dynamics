"""Train an EGNN diffusion model on crystalline Si8.

The model trained here is the one used by 04_generate_and_repaint.py. A pretrained model is provided in
references_files/pretrainedmodelSi8_epoch13.ckpt: it was trained with this script on 80 000 frames of Si8 MD
(epoch 13), instead of the 1 600 training frames of references_files/si8_database.traj used here.

Notes about this example (the chosen options):
    - Score network: Cartesian EGNN with a 5 ang radial cutoff and a smooth cutoff envelope.
    - Data: an 80/20 train/validation split of references_files/si8_database.traj (2 000 frames of Si8). The
      training frames are repeated TRAINING_DATA_REPETITIONS times per epoch, each time with a new noise.
    - Noise: exponential schedule from 1e-2 to 5 ang. The same schedule must be used for sampling.
"""

import multiprocessing
import warnings
from pathlib import Path

import ase.io
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.callbacks.checkpoint_restart import \
    restart_from_checkpoint
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.callbacks.epoch_summary_callback import \
    EpochSummaryLogger
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.ase_for_diffusion_data_module import (  # noqa
    ASEForDiffusionDataModule, ASEForDiffusionDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.loss.loss_parameters import (
    AtomTypeLossParameters, MSELossParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.scheduler import \
    ReduceLROnPlateauSchedulerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_score_network import \
    EGNNScoreNetworkParameters

# Silence noisy warnings from packages.
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
warnings.filterwarnings("ignore", message=r".*persistent_workers.*")
warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

# --- User configuration (set these for your machine and task) ---
ELEMENT_LIST = ["Si"]
NUMBER_OF_ATOMS_PER_FRAME = 8  # the (max) number of atoms per frame in TRAJECTORY_FILE_PATH
HIDDEN_DIMENSIONS_SIZE = 128
N_LAYERS = 4
WORKING_DIRECTORY = Path("run_si8")
TRAJECTORY_FILE_PATH = Path(__file__).parent / "references_files" / "si8_database.traj"
TRAINING_DATA_REPETITIONS = 10  # each epoch goes through the training frames this many times, with new noise


def main():
    """Train a new diffusion model."""
    # num_workers must be parallelized through fork instead of spawn to avoid crashing on Mac.
    multiprocessing.set_start_method("fork", force=True)
    train_new_model()


def train_new_model():
    """Train a new diffusion model."""
    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    trainer.fit(model, datamodule=data_module)
    data_module.clean_up()  # delete the HF datasets working cache now that training is done


def train_from_checkpoint():
    """Resume training from a previous checkpoint, with a new patience and learning rate."""
    # Here, we chose epoch=9.
    checkpoint_path = WORKING_DIRECTORY / "models" / "version_0" / "checkpoints" / "epoch=9.ckpt"
    new_patience = 100
    new_learning_rate = 5e-5

    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    restart_from_checkpoint(trainer, new_patience, new_learning_rate)
    trainer.fit(model, datamodule=data_module, ckpt_path=str(checkpoint_path))


def create_data_module():
    """Create the data module: an 80/20 train/validation split of TRAJECTORY_FILE_PATH."""
    train_trajectory_path, validation_trajectory_path = create_train_validation_split()
    hyper_params = ASEForDiffusionDataModuleParameters(
        data_source="ase_trajectory",
        elements=ELEMENT_LIST,
        batch_size=64,
        num_workers=4,
        max_atom=NUMBER_OF_ATOMS_PER_FRAME,
        use_fixed_lattice_parameters=True,
        noise_parameters=create_noise_parameters(),
    )
    return ASEForDiffusionDataModule(
        processed_dataset_dir=str(WORKING_DIRECTORY / "processed_data"),
        hyper_params=hyper_params,
        train_trajectory_list=[str(train_trajectory_path)],
        validation_trajectory_list=[str(validation_trajectory_path)],
        working_cache_dir=str(WORKING_DIRECTORY / "cache"),
    )


def create_train_validation_split():
    """Write an 80/20 train/validation split of TRAJECTORY_FILE_PATH, the training frames repeated."""
    frames = ase.io.read(TRAJECTORY_FILE_PATH, index=":")
    split_index = int(0.8 * len(frames))

    data_directory = WORKING_DIRECTORY / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    train_trajectory_path = data_directory / "train_conf.traj"
    validation_trajectory_path = data_directory / "validation_conf.traj"
    ase.io.write(str(train_trajectory_path), frames[:split_index] * TRAINING_DATA_REPETITIONS)
    ase.io.write(str(validation_trajectory_path), frames[split_index:])
    return train_trajectory_path, validation_trajectory_path


def create_noise_parameters():
    """Create the noise schedule shared by training and sampling."""
    return NoiseParameters(
        total_time_steps=1000, schedule_type="exponential", sigma_min_cart=1e-2, sigma_max_cart=5.,
    )


def create_diffusion_model():
    """Create the EGNN-based AXL diffusion model."""
    score_network_parameters = EGNNScoreNetworkParameters(
        num_atom_types=len(ELEMENT_LIST), n_layers=N_LAYERS,
        coordinate_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, coordinate_n_hidden_dimensions=N_LAYERS,
        message_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, message_n_hidden_dimensions=N_LAYERS,
        node_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, node_n_hidden_dimensions=N_LAYERS,
        radial_cutoff=5.,
    )
    loss_parameters = AXL(
        A=AtomTypeLossParameters(lambda_weight=0.0),
        X=MSELossParameters(lambda_weight=1.0),
        L=MSELossParameters(lambda_weight=0.0),
    )
    optimizer_parameters = OptimizerParameters(name="adamw", learning_rate=1.0e-4, weight_decay=5.0e-8)
    scheduler_parameters = ReduceLROnPlateauSchedulerParameters(factor=0.9, patience=4)

    diffusion_parameters = AXLDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=loss_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
    )
    return AXLDiffusionLightningModel(diffusion_parameters)


def create_trainer():
    """Create the trainer, with checkpointing, early stopping, TensorBoard logging and a per-epoch summary log."""
    models_directory = str(WORKING_DIRECTORY / "models")

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_epoch_loss", mode="min",
        save_top_k=-1, every_n_epochs=1, filename="{epoch}",
    )
    early_stopping_callback = EarlyStopping(monitor="validation_epoch_loss", mode="min", patience=25)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    epoch_summary_callback = EpochSummaryLogger(early_stopping_callback, WORKING_DIRECTORY / "training.log")
    logger = TensorBoardLogger(save_dir=models_directory, name="")

    return Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback, epoch_summary_callback],
        logger=logger,
        max_epochs=25,
        log_every_n_steps=1,
    )


if __name__ == "__main__":
    main()
