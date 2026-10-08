import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import (
    E_GCL, EGNN)
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import (
    unsorted_segment_mean, unsorted_segment_sum)


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(0)


@pytest.fixture()
def radial_cutoff():
    return 3.0


@pytest.fixture()
def node_features_size():
    return 5


@pytest.fixture()
def egcl_hyperparameters(node_features_size, radial_cutoff):
    return dict(
        input_size=node_features_size,
        output_size=node_features_size,
        message_n_hidden_dimensions=1,
        message_hidden_dimensions_size=8,
        node_n_hidden_dimensions=1,
        node_hidden_dimensions_size=8,
        coordinate_n_hidden_dimensions=1,
        coordinate_hidden_dimensions_size=8,
        radial_cutoff=radial_cutoff,
    )


@pytest.fixture()
def egnn_hyperparameters(radial_cutoff):
    return dict(
        input_size=3,
        num_classes=2,
        message_n_hidden_dimensions=1,
        message_hidden_dimensions_size=8,
        node_n_hidden_dimensions=1,
        node_hidden_dimensions_size=8,
        coordinate_n_hidden_dimensions=1,
        coordinate_hidden_dimensions_size=8,
        radial_cutoff=radial_cutoff,
    )


class TestCutoffEnvelope:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        return E_GCL(**egcl_hyperparameters)

    def test_envelope_values(self, egcl, radial_cutoff):
        distance = torch.tensor([0.0, 0.5 * radial_cutoff, radial_cutoff, 1.5 * radial_cutoff])
        envelope = egcl.cutoff_envelope((distance**2).unsqueeze(1))
        expected = torch.tensor([1.0, 0.5, 0.0, 0.0]).unsqueeze(1)
        torch.testing.assert_close(envelope, expected)

    def test_envelope_decreases_inside_cutoff(self, egcl, radial_cutoff):
        distance = torch.linspace(0.01, 0.99 * radial_cutoff, 100)
        envelope = egcl.cutoff_envelope((distance**2).unsqueeze(1)).squeeze(1)
        assert torch.all(envelope[1:] < envelope[:-1])
        assert torch.all(envelope > 0.0)


class TestAggregate:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        return E_GCL(**egcl_hyperparameters)

    @pytest.fixture()
    def num_nodes(self):
        return 4

    @pytest.fixture()
    def row(self):
        return torch.tensor([0, 0, 0, 1, 1, 2])

    @pytest.fixture()
    def values(self, row):
        return torch.randn(len(row), 3)

    @pytest.fixture()
    def envelope(self):
        return torch.tensor([1.0, 0.5, 0.25, 0.2, 0.0, 0.0]).unsqueeze(1)

    def test_weighted_mean(self, egcl, values, row, num_nodes, envelope):
        computed = egcl._aggregate(values, row, num_nodes, "mean", envelope)
        expected = torch.zeros(num_nodes, values.shape[1])
        for node in range(num_nodes):
            mask = row == node
            total_weight = envelope[mask].sum()
            if total_weight > 0:
                expected[node] = (envelope[mask] * values[mask]).sum(dim=0) / total_weight
        torch.testing.assert_close(computed, expected)

    def test_weighted_sum(self, egcl, values, row, num_nodes, envelope):
        computed = egcl._aggregate(values, row, num_nodes, "sum", envelope)
        expected = torch.zeros(num_nodes, values.shape[1])
        for node in range(num_nodes):
            mask = row == node
            expected[node] = (envelope[mask] * values[mask]).sum(dim=0)
        torch.testing.assert_close(computed, expected)

    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_zero_weight_nodes_are_zero(self, egcl, values, row, num_nodes, envelope, aggregation):
        computed = egcl._aggregate(values, row, num_nodes, aggregation, envelope)
        torch.testing.assert_close(computed[2:], torch.zeros(2, values.shape[1]))

    @pytest.mark.parametrize(
        "aggregation, aggregation_fn", [("mean", unsorted_segment_mean), ("sum", unsorted_segment_sum)]
    )
    def test_no_envelope_is_plain_aggregation(self, egcl, values, row, num_nodes, aggregation, aggregation_fn):
        computed = egcl._aggregate(values, row, num_nodes, aggregation, None)
        expected = aggregation_fn(values, row, num_segments=num_nodes)
        torch.testing.assert_close(computed, expected)


class TestEGCLWithoutSmoothCutoff:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters, smooth_cutoff=False)
        model.eval()
        return model

    @pytest.fixture()
    def num_nodes(self):
        return 4

    @pytest.fixture()
    def h(self, num_nodes, node_features_size):
        return torch.randn(num_nodes, node_features_size)

    @pytest.fixture()
    def coord(self, num_nodes):
        return torch.randn(num_nodes, 3)

    @pytest.fixture()
    def edge_index(self):
        pairs = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]]).float()
        shifts = torch.zeros(len(pairs), 3)
        shifts[4:, 0] = torch.tensor([1.0, -1.0])
        return torch.cat([pairs, shifts], dim=1)

    def test_forward_matches_plain_mean_aggregation(self, egcl, h, coord, edge_index, num_nodes):
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        coord_diff = coord[row] - coord[col] - edge_index[:, 2:]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        with torch.no_grad():
            messages = egcl.message_model(h[row], h[col], radial)
            expected_coord = coord + unsorted_segment_mean(
                coord_diff * egcl.coord_mlp(messages), row, num_segments=num_nodes
            )
            aggregated_messages = unsorted_segment_mean(messages, row, num_segments=num_nodes)
            expected_h = h + egcl.node_mlp(torch.cat([h, aggregated_messages], dim=1))

            computed_h, computed_coord = egcl(h, edge_index, coord.clone())

        torch.testing.assert_close(computed_h, expected_h)
        torch.testing.assert_close(computed_coord, expected_coord)


class TestEGNNContinuityAtCutoff:

    @pytest.fixture()
    def step(self):
        return 1.0e-3

    @pytest.fixture()
    def distances(self, radial_cutoff, step):
        return torch.arange(radial_cutoff - 0.2, radial_cutoff + 0.2, step)

    @pytest.fixture()
    def cell_lengths(self):
        return torch.tensor([[20.0, 20.0, 20.0]])

    @pytest.fixture()
    def h(self):
        return torch.randn(4, 3)

    def compute_outputs(self, egnn, h, cell_lengths, distances):
        outputs = []
        with torch.no_grad():
            for distance in distances:
                x = torch.tensor(
                    [
                        [5.0, 5.0, 5.0],
                        [5.0 + distance, 5.0, 5.0],
                        [5.0, 6.5, 5.0],
                        [6.5 + distance, 5.0, 5.0],
                    ]
                )
                model_outputs = egnn(h, x, cell_lengths)
                outputs.append(torch.cat([model_outputs.X.flatten(), model_outputs.A.flatten()]))
        outputs = torch.stack(outputs)
        return (outputs[1:] - outputs[:-1]).abs().max(dim=1).values

    @pytest.mark.parametrize("n_layers", [1, 2])
    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_smooth_cutoff_is_continuous(
        self, egnn_hyperparameters, h, cell_lengths, distances, step, n_layers, aggregation
    ):
        egnn = EGNN(
            **egnn_hyperparameters, n_layers=n_layers, coords_agg=aggregation, message_agg=aggregation,
            smooth_cutoff=True,
        )
        egnn.eval()
        output_changes = self.compute_outputs(egnn, h, cell_lengths, distances)
        assert output_changes.max() < 5.0 * step

    def test_hard_cutoff_is_discontinuous(self, egnn_hyperparameters, h, cell_lengths, distances):
        egnn = EGNN(**egnn_hyperparameters, n_layers=1, smooth_cutoff=False)
        egnn.eval()
        output_changes = self.compute_outputs(egnn, h, cell_lengths, distances)
        assert output_changes.max() > 2.0e-2


class TestRebuildEdgesEveryLayer:

    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def number_of_atoms(self):
        return 6

    @pytest.fixture()
    def cell_lengths(self, batch_size):
        return torch.tensor([[5.0, 5.0, 5.0], [6.0, 5.5, 5.0]])[:batch_size]

    @pytest.fixture()
    def h(self, batch_size, number_of_atoms):
        return torch.randn(batch_size * number_of_atoms, 3)

    @pytest.fixture()
    def x(self, batch_size, number_of_atoms, cell_lengths):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, 3)
        return (relative_coordinates * cell_lengths.unsqueeze(1)).reshape(-1, 3)

    @pytest.mark.parametrize("rebuild_edges_every_layer, expected_calls", [(True, 3), (False, 1)])
    def test_number_of_edge_builds(
        self, egnn_hyperparameters, h, x, cell_lengths, rebuild_edges_every_layer, expected_calls
    ):
        egnn = EGNN(**egnn_hyperparameters, n_layers=3, rebuild_edges_every_layer=rebuild_edges_every_layer)
        original_build_edges = egnn._build_edges
        calls = []

        def counting_build_edges(*args, **kwargs):
            calls.append(1)
            return original_build_edges(*args, **kwargs)

        egnn._build_edges = counting_build_edges
        egnn(h, x, cell_lengths)
        assert len(calls) == expected_calls

    def test_single_layer_is_independent_of_rebuild(self, egnn_hyperparameters, h, x, cell_lengths):
        egnn_rebuild = EGNN(**egnn_hyperparameters, n_layers=1, rebuild_edges_every_layer=True)
        egnn_once = EGNN(**egnn_hyperparameters, n_layers=1, rebuild_edges_every_layer=False)
        egnn_once.load_state_dict(egnn_rebuild.state_dict())
        egnn_rebuild.eval()
        egnn_once.eval()
        with torch.no_grad():
            outputs_rebuild = egnn_rebuild(h, x.clone(), cell_lengths)
            outputs_once = egnn_once(h, x.clone(), cell_lengths)
        torch.testing.assert_close(outputs_rebuild.X, outputs_once.X)
        torch.testing.assert_close(outputs_rebuild.A, outputs_once.A)


@pytest.mark.parametrize("smooth_cutoff", [True, False])
@pytest.mark.parametrize("rebuild_edges_every_layer", [True, False])
def test_score_network_passes_options_to_egnn(smooth_cutoff, rebuild_edges_every_layer):
    score_network_parameters = EGNNScoreNetworkParameters(
        radial_cutoff=3.0,
        num_atom_types=1,
        smooth_cutoff=smooth_cutoff,
        rebuild_edges_every_layer=rebuild_edges_every_layer,
    )
    score_network = EGNNScoreNetwork(score_network_parameters)
    assert score_network.egnn.rebuild_edges_every_layer == rebuild_edges_every_layer
    for graph_layer in score_network.egnn.graph_layers:
        assert graph_layer.smooth_cutoff == smooth_cutoff
        assert graph_layer.radial_cutoff == 3.0


def test_score_network_options_default_to_true():
    score_network_parameters = EGNNScoreNetworkParameters(radial_cutoff=3.0, num_atom_types=1)
    assert score_network_parameters.smooth_cutoff
    assert score_network_parameters.rebuild_edges_every_layer
