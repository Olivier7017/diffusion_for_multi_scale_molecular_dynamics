"""EGNN on cartesian positions, with periodic edges built from the positions."""

from typing import Optional

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import \
    EGNN
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import \
    build_graph_periodic_edges_batch


class CartesianEGNN(EGNN):
    """EGNN on cartesian positions, with periodic edges built from the positions."""

    def __init__(self, rebuild_edges_every_layer: bool = True, **egnn_arguments):
        """Init method.

        Args:
            rebuild_edges_every_layer: if True, rebuild the periodic edges from the layer-updated positions before
                every layer. If False, build them once from the input positions. Defaults to True.
            egnn_arguments: the arguments of EGNN.
        """
        super().__init__(**egnn_arguments)
        self.rebuild_edges_every_layer = rebuild_edges_every_layer

    def _build_edges(
        self,
        x: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build the periodic edges within radial_cutoff.

        Args:
            x: node coordinates. size is number of nodes, spatial dimension
            cell_lengths: per-sample orthogonal cell side lengths. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            edges as [i, j, shift_x, shift_y, shift_z]. size: number of edges, 2 + spatial_dimension
        """
        batch_size = cell_lengths.shape[0]
        batched_x = einops.rearrange(
            x, "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size, natoms=x.shape[0] // batch_size,
        )
        return build_graph_periodic_edges_batch(
            batched_x, cell_lengths, self.radial_cutoff, natoms=natoms,
        )

    @staticmethod
    def _get_edge_distances_squared(edges: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get the distance squared of every periodic edge.

        Args:
            edges: edges as [i, j, shift_x, shift_y, shift_z]. size: number of edges, 2 + spatial_dimension
            x: node coordinates. size is number of nodes, spatial dimension

        Returns:
            distance squared of every edge. size: number of edges, 1
        """
        row, col = edges[:, 0].long(), edges[:, 1].long()
        return ((x[row] - x[col] - edges[:, 2:]) ** 2).sum(dim=1, keepdim=True)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
    ) -> AXL:
        """Forward instructions for the model.

        Args:
            h: node features. size is number of nodes (atoms), input size
            x: cartesian node coordinates. size is number of nodes, spatial dimension
            cell_lengths: per-sample orthogonal cell side lengths, used to build the periodic
                edges. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            estimated score in an AXL namedtuple (see EGNN._get_model_outputs).
        """
        h = self.embedding_in(h)
        edges = self._build_edges(x, cell_lengths, natoms)
        for layer_index, graph_layer in enumerate(self.graph_layers):
            if self.rebuild_edges_every_layer and layer_index > 0:
                edges = self._build_edges(x, cell_lengths, natoms)
            h, x = graph_layer(h, edges, x, self._get_edge_distances_squared(edges, x))
        return self._get_model_outputs(h, x)
