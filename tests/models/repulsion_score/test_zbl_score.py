from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.repulsion_score.zbl_score import (
    ZBLRepulsionScore,
)


def _tests_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == "tests":
            return p
    raise RuntimeError("Could not locate tests/ directory from __file__.")


def read_lammps_forces(fn):
    """Read the forces contained in data/models/repulsion_score."""
    with open(fn, "r") as f:
        lines = f.readlines()
    forces = []
    start_read = False
    for line in lines:
        if line.strip().startswith("ITEM: ATOMS"):
            start_read = True
            continue
        if start_read:
            forces.append([float(x) for x in line.split()[-3:]])
    return np.array(forces)


def get_pos_and_basis_vectors_sige():
    """The positions and cell used in the precomputed LAMMPS simulation."""
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
            [5.9723, 1.1048, 6.9392],
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
    vec = [[[7.2200, 0.0000, 0.0000],
            [0.0000, 7.2200, 0.0000],
            [0.0000, 0.0000, 7.2200]]]
    return pos, vec


def test_zbl_forces_match_lammps_sige():
    """Test the forces of ZBLRepulsionScore with precomputed ones from LAMMPS."""
    repulsion_score = ZBLRepulsionScore(
        cutoff_radius=2.19293,
        inner_radius_fraction=0.5552844824048191,
        element_list=["Si", "Ge"],
        device="cpu",
    )
    A = torch.tensor([[2, 1] * 16], dtype=torch.long)
    pos, vec = get_pos_and_basis_vectors_sige()
    cartesian_positions = torch.tensor(pos)
    basis_vectors = torch.tensor(vec)

    torch_forces = repulsion_score.get_forces(A, cartesian_positions, basis_vectors)[0].detach().cpu().numpy()
    dump_path = _tests_root() / "data" / "models" / "repulsion_score" / "SiGe.dump"
    lammps_forces = read_lammps_forces(dump_path)

    assert torch_forces.shape == lammps_forces.shape == (32, 3)
    np.testing.assert_allclose(torch_forces, lammps_forces, rtol=1e-2, atol=1e-6)
