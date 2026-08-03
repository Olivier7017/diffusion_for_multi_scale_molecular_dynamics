import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.logging import (
    clean_up_campaign_logger, set_up_campaign_logger)
from diffusion_for_multi_scale_molecular_dynamics.calc.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.io.artn import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields_from_dump
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.base_sample_maker import \
    BaseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.namespace import (
    AXL_STRUCTURE_IN_NEW_BOX, AXL_STRUCTURE_IN_ORIGINAL_BOX)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_converter import \
    StructureConverter


class ActiveLearning:
    """Active Learning.

    This class is the main driver of the active learning loop, dispatching sub-tasks as needed.

    Active learning flows as follows:
        - start with a MLIP that has been pretrained (ie, is not completely empty)
        - Iterate until SUCCESS:
            * deploy the MLIP
            * run artn with the MLIP:
                - SUCCESS if no encountered structure has an uncertainty above the threshold; exit.
                - INTERRUPTION otherwise (an uncertain structure was found).
            * collect the uncertain structure
            * use the Oracle to evaluate the uncertain structure
            * add the uncertain structure to the MLIP's training database
            * retrain the MLIP
    """

    def __init__(
        self,
        oracle_single_point_calculator: BaseSinglePointCalculator,
        sample_maker: BaseSampleMaker,
        artn_driver: ArtnDriver,
    ):
        """Init method.

        Args:
            oracle_single_point_calculator: class responsible for generating of ground truth labels.
            sample_maker: class responsible for generating samples for active learning.
            artn_driver: class responsible for running LAMMPS + ARTn.
        """
        self.oracle_calculator = oracle_single_point_calculator
        self.sample_maker = sample_maker
        self.artn_driver = artn_driver
        self._structure_converter = StructureConverter(list_of_element_symbols=sample_maker.arguments.element_list)

    def _get_uncertain_structure_and_uncertainties(
        self, artn_working_directory: Path
    ) -> Tuple[Structure, np.ndarray]:
        """Get uncertain structure.

        This method assumes the CONVENTION that the ARTn + LAMMPS run will produce a file
        named 'uncertain_dump.yaml' that contains the uncertain structure.
        """
        lammps_dump_path = artn_working_directory / "uncertain_dump.yaml"
        assert lammps_dump_path.is_file(), f"The file {lammps_dump_path} is missing."

        list_structures, _, _, list_uncertainties = extract_all_fields_from_dump(
            lammps_dump_path
        )
        uncertain_structure = list_structures[0]
        uncertainties = list_uncertainties[0]
        return uncertain_structure, uncertainties

    def _make_samples(
        self, structure: Structure, uncertainty_per_atom: np.ndarray
    ) -> Tuple[List[Structure], List[np.array], List[Dict[str, Any]]]:
        """Make samples.

        This method handles the back-and-forth transformation from Pymatgen Structures to AXL structures.

        Ars:
            structure: Pymatgen structure to make samples from.
            uncertainty_per_atom: uncertainty per atom.

        Returns:
            list_sample_structures: list of sampled structures.
            list_active_indices: The indices of the active atoms in the sample structures.
            list_additional_information: list of additional information.
        """
        axl_structure = self._structure_converter.convert_structure_to_axl(structure)
        (list_sample_axl_structures,
         list_active_indices,
         list_sample_additional_information) = (
            self.sample_maker.make_samples(axl_structure, uncertainty_per_atom)
        )

        list_sample_structures = [
            self._structure_converter.convert_axl_to_structure(axl_structure)
            for axl_structure in list_sample_axl_structures
        ]
        converted_list_additional_information = [
            self._convert_axl_to_structure_in_dict(sample_info)
            for sample_info in list_sample_additional_information
        ]
        return (
            list_sample_structures,
            list_active_indices,
            converted_list_additional_information,
        )

    def _convert_axl_to_structure_in_dict(
        self, sample_additional_information: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert AXL elements of an additional information dictionary to pymatgen structure.

        Args:
            sample_additional_information: additional information about a sample in a dictionary.

        Returns:
            new_structures: additional information about a sample in a dictionary with AXL as Structure
        """
        converted_info = {}
        for k, v in sample_additional_information.items():
            if k in [AXL_STRUCTURE_IN_ORIGINAL_BOX, AXL_STRUCTURE_IN_NEW_BOX]:
                converted_info[k] = self._structure_converter.convert_axl_to_structure(v)
            else:
                converted_info[k] = v
        return converted_info

    def _convert_single_point_calculations_to_dataframe(
        self,
        list_single_point_calculations: List[SinglePointCalculation],
        list_sample_information: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert single point calculations to dataframe."""
        rows = []
        for calculation, sample_information in zip(
            list_single_point_calculations, list_sample_information
        ):

            constrained_indices = sample_information["constrained_atom_indices"]
            structure = calculation.structure
            constraint_mask = np.zeros(len(structure), dtype=int)
            constraint_mask[constrained_indices] = 1
            structure.add_site_property('constrained', constraint_mask)
            structure.add_site_property('forces', calculation.forces)

            row = dict(
                calculation_type=calculation.calculation_type,
                structure=structure,
                energy=calculation.energy,
            )
            rows.append(row)

        df = pd.DataFrame(data=rows)
        return df

    def _log_campaign_details(self, campaign_working_directory_path: Path, campaign_details: Dict):
        """Log campaign details."""
        output_file = campaign_working_directory_path / "campaign_details.yaml"
        with open(str(output_file), "w") as fd:
            yaml.dump(campaign_details, fd)

    def run_campaign(
        self,
        uncertainty_threshold: float,
        mlip: BaseMLIP,
        working_directory: Path,
        maximum_number_of_rounds: int = 100,
    ):
        """Run campaign.

        Perform a full campaign of active learning.

        Args:
            uncertainty_threshold: the uncertainty threshold to interrupt an ARTn run.
            mlip: the machine-learning interatomic potential to drive and refine. It is assumed to be
                already pretrained, so that it can be deployed and run from the first round.
            working_directory: top directory where all the various artifacts from this campaign will be written.
            maximum_number_of_rounds: maximum number of active learning rounds. This is useful to avoid
                infinite loops...
        """
        working_directory.mkdir(parents=True, exist_ok=True)
        logger = set_up_campaign_logger(working_directory)
        logger.info("Starting Active Learning Simulation")

        mlip.prepare_mlip_first_round(working_directory / "initial_mlip")

        round_number = 0
        while round_number <= maximum_number_of_rounds:
            round_number += 1
            logger.info(f"Starting Round {round_number}")

            campaign_is_complete = self._run_one_campaign_iteration(
                round_number=round_number,
                uncertainty_threshold=uncertainty_threshold,
                mlip=mlip,
                working_directory=working_directory,
                logger=logger,
            )
            if campaign_is_complete:
                break

        campaign_details = dict(uncertainty_threshold=float(uncertainty_threshold),
                                final_round=int(round_number),
                                **mlip.training_metrics())
        self._log_campaign_details(campaign_working_directory_path=working_directory,
                                   campaign_details=campaign_details)
        # Delete the logger to avoid overlogging across campaigns.
        clean_up_campaign_logger(logger)

    def _run_one_campaign_iteration(
        self,
        round_number: int,
        uncertainty_threshold: float,
        mlip: BaseMLIP,
        working_directory: Path,
        logger,
    ) -> bool:
        """Run a single active learning round; return True when the campaign is complete.

        The round is made of the following steps:
            1. Run ARTn with the MLIP.
            2. Extract the uncertainty per atom.
            3. Excise environments and repaint samples.
            4. Evaluate the repainted samples with the Oracle.
            5. Add the labelled structures to the training database.
            6. Retrain the MLIP.
        """
        current_sub_directory = working_directory / f"round_{round_number}"

        # 1. Run ARTn with the MLIP.
        # The artn_driver will create this directory.
        artn_working_directory = current_sub_directory / "lammps_artn"
        logger.info("  Launching ARTn simulation...")
        calculation_state = self.artn_driver.run(
            mlip=mlip,
            working_directory=artn_working_directory,
            uncertainty_threshold=uncertainty_threshold,
        )
        logger.info(f"  ARTn state is {calculation_state}")

        if calculation_state == CalculationState.SUCCESS:
            logger.info("Active Learning Campaign is Complete. Exiting.")
            return True

        # 2. Extract the uncertainty per atom.
        logger.info("  Extracting uncertain structure from ARTn work directory...")
        uncertain_structure, uncertainty_per_atom = (
            self._get_uncertain_structure_and_uncertainties(artn_working_directory)
        )

        number_of_uncertain_envs = np.sum(uncertainty_per_atom > uncertainty_threshold)
        logger.info(
            f" -> There are {number_of_uncertain_envs} environments with uncertainty above the threshold."
        )

        # 3. Excise environments and repaint samples.
        logger.info("  Making new samples based on uncertainties.")
        list_sample_structures, list_active_indices, list_sample_information = (
            self._make_samples(uncertain_structure, uncertainty_per_atom))

        # 4. Evaluate the repainted samples with the Oracle.
        logger.info("  Labelling samples with oracle...")
        oracle_directory = current_sub_directory / "oracle"
        oracle_directory.mkdir(parents=True, exist_ok=True)

        time1 = time.time()
        list_single_point_calculations = []
        for idx, structure in enumerate(list_sample_structures):
            results_path = oracle_directory / f"dump_{idx}.yaml"
            result = self.oracle_calculator.calculate(structure, results_path=results_path)
            list_single_point_calculations.append(result)
        time2 = time.time()
        logger.info(
            f" -> It took {time2 - time1: 6.2e} seconds to compute labels with Oracle."
        )

        logger.info("  Converting labelled samples and writing pickle to disk.")
        oracle_df = self._convert_single_point_calculations_to_dataframe(
            list_single_point_calculations, list_sample_information
        )
        output_file = oracle_directory / "oracle_single_point_calculations.pkl"
        oracle_df.to_pickle(output_file)

        # 5. Add the labelled structures to the training database.
        logger.info("  Adding labelled samples to the MLIP training database.")
        for single_point_calculation, active_environment_indices \
                in zip(list_single_point_calculations, list_active_indices):
            mlip.add_labelled_structure(
                single_point_calculation,
                active_environment_indices=active_environment_indices,
            )

        # 6. Retrain the MLIP.
        logger.info("  Retraining the MLIP...")
        mlip_training_directory = current_sub_directory / "mlip_training"

        mlip.train(mlip_training_directory)
        self._update_latest_mlip_symlink(working_directory, mlip_training_directory)

        mlip.write_logger_info(logger)
        return False

    @staticmethod
    def _update_latest_mlip_symlink(working_directory: Path, mlip_training_directory: Path):
        """Point 'latest_mlip' at the most recent MLIP training directory."""
        symlink_path = working_directory / "latest_mlip"
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(mlip_training_directory.resolve(), target_is_directory=True)
