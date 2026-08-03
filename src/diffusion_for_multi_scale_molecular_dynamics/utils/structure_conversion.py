"""Normalize configuration objects into pymatgen Structures."""

from typing import Union

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def to_pymatgen_structure(configuration: Union[Structure, Atoms]) -> Structure:
    """Convert a configuration (pymatgen Structure or ase.Atoms) to a pymatgen Structure."""
    if isinstance(configuration, Structure):
        return configuration
    if isinstance(configuration, Atoms):
        return AseAtomsAdaptor.get_structure(configuration)
    raise TypeError(
        f"Cannot convert a configuration of type '{type(configuration).__name__}' to a Structure; "
        "expected a pymatgen Structure or ase.Atoms."
    )
