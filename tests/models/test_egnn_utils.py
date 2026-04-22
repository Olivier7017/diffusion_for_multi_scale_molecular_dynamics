import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.egnn_utils import (
    get_edges_batch, get_edges_with_radial_cutoff, unsorted_segment_mean,
    unsorted_segment_sum)


@pytest.fixture()
def num_messages():
    return 15


@pytest.fixture()
def num_ids():
    return 3


@pytest.fixture()
def message_ids(num_messages, num_ids):
    return torch.randint(low=0, high=num_ids, size=(num_messages,))


@pytest.fixture()
def num_message_features():
    return 2


@pytest.fixture()
def messages(num_messages, num_message_features):
    return torch.randn(num_messages, num_message_features)


SI_BOND_LENGTH_ANG = 2.36


@pytest.mark.parametrize("box_size", [10.0, 20.0])
def test_get_edges_batch_distances_are_cell_size_independent(box_size):
    """Fully-connected edge distances should be the same regardless of box size."""
    reduced_coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                         [SI_BOND_LENGTH_ANG / box_size, 0.0, 0.0]]])
    unit_cell = torch.diag(torch.tensor([box_size, box_size, box_size])).unsqueeze(0)

    edges = get_edges_batch(n_nodes=2, batch_size=1,
                            reduced_coordinates=reduced_coordinates,
                            unit_cell=unit_cell)

    distances = edges[:, 2]
    torch.testing.assert_close(distances, torch.full_like(distances, SI_BOND_LENGTH_ANG))


@pytest.mark.parametrize("box_size", [10.0, 20.0])
def test_get_edges_with_radial_cutoff_distances_are_cell_size_independent(box_size):
    """Radial-cutoff edge distances should be the same regardless of box size."""
    radial_cutoff = 2.5
    reduced_coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                         [SI_BOND_LENGTH_ANG / box_size, 0.0, 0.0]]])
    unit_cell = torch.diag(torch.tensor([box_size, box_size, box_size])).unsqueeze(0)

    edges = get_edges_with_radial_cutoff(reduced_coordinates, unit_cell,
                                         radial_cutoff=radial_cutoff, spatial_dimension=3)

    distances = edges[:, 2]
    torch.testing.assert_close(distances, torch.full_like(distances, SI_BOND_LENGTH_ANG))


def test_unsorted_segment_sum(
    num_messages, num_ids, message_ids, num_message_features, messages
):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message

    message_summed = unsorted_segment_sum(messages, message_ids, num_ids)
    assert message_summed.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_summed, expected_message_sums)


def test_unsorted_segment_mean(
    num_messages, num_ids, message_ids, num_message_features, messages
):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    expected_counts = torch.zeros(num_ids, 1)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message
        expected_counts[m_id] += 1
    expected_message_average = expected_message_sums / torch.maximum(
        expected_counts, torch.ones_like(expected_counts)
    )

    message_averaged = unsorted_segment_mean(messages, message_ids, num_ids)
    assert message_averaged.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_averaged, expected_message_average)
