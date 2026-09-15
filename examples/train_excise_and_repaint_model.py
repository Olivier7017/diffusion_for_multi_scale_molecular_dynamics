"""Train a EGNN to repaint a-Si.

This can be used to generate a model to run the examples which relies on a trained model.

Notes about this example (the chosen options):
    - Score network: EGNN with a 5 ang radial cutoff
    - Data: an 80/20 train/validation split.
"""

import multiprocessing
import warnings
from pathlib import Path

# num_workers must be parallelized through fork instead of spawn to avoid crashing on Mac.
multiprocessing.set_start_method("fork", force=True)

# Silence noisy warnings from packages.
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
warnings.filterwarnings("ignore", message=r".*persistent_workers.*")
warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated.*")

import ase.io
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.callbacks.epoch_summary_callback import \
    EpochSummaryLogger
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.ase_for_diffusion_data_module import (
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

# --- User configuration (set these for your machine and task) ---
ELEMENT_LIST = ["Si"]
NUMBER_OF_ATOMS_PER_FRAME = 1000  # the (max) number of atoms per frame in TRAJECTORY_FILE_PATH
WORKING_DIRECTORY = Path("run")
TRAJECTORY_FILE_PATH = Path(__file__).parent / "references_files" / "aSi_200conf.traj"


def main():
    """Train the diffusion model."""
    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    trainer.fit(model, datamodule=data_module)
    data_module.clean_up()  # delete the HF datasets working cache now that training is done


def restart_from_checkpoint():
    """Resume training from a previous checkpoint."""
    # Here, we chose epoch=9.
    checkpoint_path = WORKING_DIRECTORY / "models" / "version_0" / "checkpoints" / "epoch=9.ckpt"
    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    trainer.fit(model, datamodule=data_module, ckpt_path=str(checkpoint_path))


def create_data_module():
    """Create the data module: an 80/20 train/validation split of TRAJECTORY_FILE_PATH."""
    train_trajectory_path, validation_trajectory_path = create_train_validation_split()
    hyper_params = ASEForDiffusionDataModuleParameters(
        data_source="ase_trajectory",
        elements=ELEMENT_LIST,
        batch_size=4,
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
    """Write an 80/20 train/validation split of TRAJECTORY_FILE_PATH."""
    frames = ase.io.read(TRAJECTORY_FILE_PATH, index=":")
    split_index = int(0.8 * len(frames))

    data_directory = WORKING_DIRECTORY / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    train_trajectory_path = data_directory / "train_conf.traj"
    validation_trajectory_path = data_directory / "validation_conf.traj"
    ase.io.write(str(train_trajectory_path), frames[:split_index])
    ase.io.write(str(validation_trajectory_path), frames[split_index:])
    return train_trajectory_path, validation_trajectory_path


def create_noise_parameters():
    """Create the noise schedule shared by training and (later) sampling."""
    return NoiseParameters(
        total_time_steps=1000, schedule_type="exponential", sigma_min_cart=5e-5, sigma_max_cart=5.,
    )


def create_diffusion_model():
    """Create the EGNN-based AXL diffusion model."""
    score_network_parameters = EGNNScoreNetworkParameters(
        num_atom_types=len(ELEMENT_LIST), n_layers=4,
        coordinate_hidden_dimensions_size=64, coordinate_n_hidden_dimensions=4,
        message_hidden_dimensions_size=64, message_n_hidden_dimensions=4,
        node_hidden_dimensions_size=64, node_n_hidden_dimensions=4,
        edges="radial_cutoff", radial_cutoff=5.0,
    )
    loss_parameters = AXL(
        A=AtomTypeLossParameters(lambda_weight=0.0),
        X=MSELossParameters(lambda_weight=1.0),
        L=MSELossParameters(lambda_weight=0.0),
    )
    optimizer_parameters = OptimizerParameters(name="adamw", learning_rate=1.0e-4, weight_decay=5.0e-8)
    scheduler_parameters = ReduceLROnPlateauSchedulerParameters(factor=0.9, patience=5)

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
        max_epochs=100,
        log_every_n_steps=1,
    )


if __name__ == "__main__":
    main()
