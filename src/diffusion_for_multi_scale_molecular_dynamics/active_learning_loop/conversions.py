"""Conversions used by the active learning loop: sample AXL structures and oracle results to shareable forms."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.namespace import (
    AXL_STRUCTURE_IN_NEW_BOX, AXL_STRUCTURE_IN_ORIGINAL_BOX)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_converter import \
    StructureConverter


def convert_axl_to_structure_in_dict(
    sample_additional_information: Dict[str, Any], structure_converter: StructureConverter,
) -> Dict[str, Any]:
    """Convert the AXL entries of a sample's additional-information dictionary to pymatgen structures."""
    converted_info = {}
    for key, value in sample_additional_information.items():
        if key in [AXL_STRUCTURE_IN_ORIGINAL_BOX, AXL_STRUCTURE_IN_NEW_BOX]:
            converted_info[key] = structure_converter.convert_axl_to_structure(value)
        else:
            converted_info[key] = value
    return converted_info


def convert_single_point_calculations_to_dataframe(
    list_single_point_calculations: List[SinglePointCalculation],
    list_sample_information: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Convert single point calculations (with their sample information) to a dataframe of labelled structures."""
    rows = []
    for calculation, sample_information in zip(list_single_point_calculations, list_sample_information):
        constrained_indices = sample_information["constrained_atom_indices"]
        structure = calculation.structure
        constraint_mask = np.zeros(len(structure), dtype=int)
        constraint_mask[constrained_indices] = 1
        structure.add_site_property('constrained', constraint_mask)
        structure.add_site_property('forces', calculation.forces)

        rows.append(dict(
            calculation_type=calculation.calculation_type,
            structure=structure,
            energy=calculation.energy,
        ))
    return pd.DataFrame(data=rows)
