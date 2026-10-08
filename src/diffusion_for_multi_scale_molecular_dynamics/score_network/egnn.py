"""Equivariant Graph Neural Network (EGNN).

This implementation is based on the following link:
https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L106

It implements EGNN as described in the paper "E(n) Equivariant Graph Neural Networks".

The file is modified from the original download to fit our own linting style and add additional controls.
"""

from typing import Callable, Optional, Tuple

import einops
import torch
from torch import nn

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import (
    build_graph_periodic_edges_batch, unsorted_segment_mean, unsorted_segment_sum)


class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        radial_cutoff: float,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = "mean",
        message_agg: str = "mean",
        tanh: bool = False,
        smooth_cutoff: bool = True,
    ):
        """E_GCL layer initialization.

        Args:
            input_size: number of node features in the input
            output_size: number of node features in the output
            message_n_hidden_dimensions: number of hidden layers of the message (edge) MLP
            message_hidden_dimensions_size: size of the hidden layers of the message (edge) MLP
            node_n_hidden_dimensions: number of hidden layers of the node update MLP
            node_hidden_dimensions_size: size of the hidden layers of the node update MLP
            coordinate_n_hidden_dimensions: number of hidden layers of the coordinate update MLP
            coordinate_hidden_dimensions_size: size of the hidden layers of the coordinate update MLP
            radial_cutoff: cutoff distance, in Angstrom, at which the smooth cutoff envelope reaches zero.
            act_fn: activation function used in the MLPs. Defaults to nn.SiLU()
            residual: if True, add a skip connection in the nodes update. Defaults to True.
            attention: if True, multiply the message output by a gated value of the output. Defaults to False.
            normalize: if True, use a normalized version of the coordinates update i.e. x_i^l - x_j^l would be a unit
                vector in eq. 4 in https://arxiv.org/pdf/2102.09844. Defaults to False.
            coords_agg: Use a weighted mean (due to smooth cutoff) or sum aggregation for the coordinates update.
                Defaults to mean.
            message_agg: Use a weighted mean (due to smooth cutoff) or sum aggregation for the messages.
                Defaults to mean.
            tanh: if True, add a tanh non-linearity after the coordinates update. Defaults to False.
            smooth_cutoff: if True, add a cosine envelope that smoothly goes to zero at radial cutoff :
                f(r) = 0.5 * (cos(pi * r / radial_cutoff) + 1). Defaults to True.
        """
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.epsilon = 1e-8
        self.radial_cutoff = radial_cutoff
        self.smooth_cutoff = smooth_cutoff

        if coords_agg not in ["mean", "sum"]:
            raise ValueError(f"coords_agg should be mean or sum. Got {coords_agg}")
        self.coords_agg = coords_agg
        self.message_agg = message_agg

        # message update MLP i.e. message m_{ij} used in the graph neural network.
        # \phi_e is eq. (3) in https://arxiv.org/pdf/2102.09844
        # Input is a concatenation of the two node features and distance
        message_input_size = input_size * 2 + 1
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, message_hidden_dimensions_size), act_fn
        )
        for _ in range(message_n_hidden_dimensions):
            self.message_mlp.append(
                nn.Linear(
                    message_hidden_dimensions_size, message_hidden_dimensions_size
                )
            )
            self.message_mlp.append(act_fn)

        # node update mlp. Input is the node feature (size input_size) and the aggregated messages from neighbors
        # size (message_hidden_dimension)
        # \phi_h in eq. (6) in https://arxiv.org/pdf/2102.09844
        node_input_size = input_size + message_hidden_dimensions_size
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_size, node_hidden_dimensions_size), act_fn
        )
        for _ in range(node_n_hidden_dimensions):
            self.node_mlp.append(
                nn.Linear(node_hidden_dimensions_size, node_hidden_dimensions_size)
            )
            self.node_mlp.append(act_fn)
        self.node_mlp.append(nn.Linear(node_hidden_dimensions_size, output_size))

        # coordinate (x) update MLP. Input is the message m_{ij}
        # \phi_x in eq.(4) in https://arxiv.org/pdf/2102.09844

        coordinate_input_size = message_hidden_dimensions_size
        self.coord_mlp = nn.Sequential(
            nn.Linear(coordinate_input_size, coordinate_hidden_dimensions_size)
        )
        self.coord_mlp.append(act_fn)
        for _ in range(coordinate_n_hidden_dimensions):
            self.coord_mlp.append(
                nn.Linear(
                    coordinate_hidden_dimensions_size, coordinate_hidden_dimensions_size
                )
            )
            self.coord_mlp.append(act_fn)
        final_coordinate_layer = nn.Linear(
            coordinate_hidden_dimensions_size, 1, bias=False
        )
        # based on the original implementation - multiply the random initialization by 0.001 (default is 1)
        #torch.nn.init.xavier_uniform_(final_coordinate_layer.weight, gain=0.001)
        self.coord_mlp.append(final_coordinate_layer)  # initialized with a different
        if self.tanh:  # optional, add a tanh to saturate the messages
            self.coord_mlp.append(nn.Tanh())

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(message_hidden_dimensions_size, 1), nn.Sigmoid()
            )

    def message_model(
        self, source: torch.Tensor, target: torch.Tensor, radial: torch.Tensor
    ) -> torch.Tensor:
        r"""Constructs the message m_{ij} from source (j) to target (i).

        .. math::

            m_ij = \phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_{ij)}

        with :math:`a_{ij}` the edge attributes

        Args:
            source: source node features (size: number of edges, input_size)
            target: target node features (size: number of edges, input_size)
            radial: distance squared between nodes i and j (size: number of edges, 1)

        Returns:
            messages :math:`m_{ij}` size: number of edges, message_hidden_dimensions_size
        """
        out = torch.cat([source, target, radial], dim=1)
        out = self.message_mlp(out)
        if self.attention:  # gate the message by itself - optional
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def cutoff_envelope(self, radial: torch.Tensor) -> torch.Tensor:
        """Cosine cutoff envelope f(r) = 0.5 * (cos(pi * r / radial_cutoff) + 1) for r < radial_cutoff, 0 beyond.

        Args:
            radial: distance squared between nodes i and j. size: number of edges, 1

        Returns:
            envelope value of every edge, 1 at r = 0 and 0 at r = radial_cutoff. size: number of edges, 1
        """
        distance = torch.sqrt(radial + self.epsilon**2)
        envelope = 0.5 * (torch.cos(torch.pi * distance / self.radial_cutoff) + 1.0)
        return envelope * (distance < self.radial_cutoff)

    def _aggregate(
        self,
        values: torch.Tensor,
        row: torch.Tensor,
        num_segments: int,
        aggregation: str,
        envelope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate edge values onto their target node.

        Args:
            values: one value per edge. size: number of edges, feature size
            row: target node of every edge. size: number of edges
            num_segments: number of nodes
            aggregation: "mean" or "sum"
            envelope: cutoff envelope of every edge (size: number of edges, 1), or None for no smooth cutoff.

        Returns:
            aggregated values. size: number of nodes, feature size. With an envelope, "mean" is the
            envelope-weighted mean sum(f v) / sum(f) and "sum" is sum(f v); a node without neighbours gets zero.
        """
        if envelope is None:
            aggregation_fn = unsorted_segment_sum if aggregation == "sum" else unsorted_segment_mean
            return aggregation_fn(values, row, num_segments=num_segments)
        weighted_sum = unsorted_segment_sum(values * envelope, row, num_segments=num_segments)
        if aggregation == "sum":
            return weighted_sum
        total_weight = unsorted_segment_sum(envelope, row, num_segments=num_segments)
        return weighted_sum / total_weight.clamp(min=self.epsilon)

    def node_model(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        messages: torch.Tensor,
        envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Update the node features.

        .. math::

            h_i^{(l+1)} = \phi_h (h_i^l, m_i)

        Args:
            x: node features. Size number of nodes, input_size
            edge_index: source and target indices defining the edges. size: number of edges, 2
            messages: messages between nodes. size: number of edges, message_hidden_dimension_size
            envelope: cutoff envelope of every edge (size: number of edges, 1), or None for no smooth cutoff.

        Returns:
            updated node features. size: number of nodes, output_size
        """
        row = edge_index[:, 0].long()
        agg = self._aggregate(messages, row, x.size(0), self.message_agg, envelope)
        agg = torch.cat([x, agg], dim=1)  # concat h_i and m_i
        out = self.node_mlp(agg)
        if self.residual:  # optional skip connection
            out = x + out
        return out

    def coord_model(
        self,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        messages: torch.Tensor,
        envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Update the coordinates.

        .. math::

            x_i^{(l+1)} = x_i^l + C \sum_{i \neq j} (x_i - x_j) \phi_x(m_{ij})

        Args:
            coord: coordinates. size: number of nodes, spatial dimension
            edge_index: edge indices. size: number of edges, 2
            coord_diff: difference between coordinates, :math:`x_i - x_j`. size: number of edges, spatial dimension
            messages: messages between nodes i and j.  size: number of edges, message_hidden_dimensions_size
            envelope: cutoff envelope of every edge (size: number of edges, 1), or None for no smooth cutoff.

        Returns:
            updates coordinates. size: number of nodes, spatial dimension
        """
        row = edge_index[:, 0].long()
        trans = coord_diff * self.coord_mlp(messages)  # (x_i  - x_j) *  \phi_m(m_{ij})
        agg = self._aggregate(trans, row, coord.size(0), self.coords_agg, envelope)
        coord += agg
        return coord

    def coord2radial(
        self,
        edge_index: torch.Tensor,
        coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distances between linked nodes.

        Args:
            edge_index: source and destination indices, and their periodic shift (Angstrom).
                size: number of edges, 2 + spatial_dimension
            coord: coordinates. size: number of nodes, spatial dimension

        Returns:
            distance squared between nodes. size: number of edges
            distance vector between nodes. size: number of edges, spatial dimension
        """
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        shift = edge_index[:, 2:]
        coord_diff = coord[row] - coord[col] - shift
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:  # normalize distance vector to be unit vector
            # norm is detached from gradient in the original implementation - not clear why
            coord_diff = self.normalize_radial_norm(radial) * coord_diff

        return radial, coord_diff

    def normalize_radial_norm(self, radial_norm_squared: torch.Tensor) -> torch.Tensor:
        """Normalize radial distance.

        This method normalizes the distance between two nodes, insuring that the normalized
        distance vector smoothly goes to zero when the nodes overlap and goes to a unit length when the
        nodes are far apart. This insures that the model as a whole is smooth.

        Args:
            radial_norm_squared: the square distance between two nodes.

        Returns:
            normalized_radial_norm: normalized distance between two nodes.
        """
        # This function computes a normalization for a radial distance.
        # For r the vector radial distance, define
        #
        #   norm_r = f(|r|^2) * r
        #
        # where f(.) is this function. We have that
        # - |norm_r| goes to zero like |r|^2 as |r|-> 0
        # - |norm_r| goes to one  as |r|-> infty
        #
        # In effect, this function creates a "hole" around |r| = 0 to insure that there are no
        # discontinuous jumps in norm_r.
        return torch.tanh(radial_norm_squared) / torch.sqrt(radial_norm_squared + self.epsilon**2)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings and coordinates.

        Args:
            h: node features. size: number of nodes, input_size
            edge_index: source and destination indices, and their periodic shift (Angstrom).
                size: number of edges, 2 + spatial_dimension
            coord: node positions. size: number of nodes, spatial dimension

        Returns:
            updated node features. size: number of nodes, output_size
            updated coordinates. size: number of nodes, spatial dimension
        """
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        # compute cartesian distances between nodes (atoms)
        radial, coord_diff = self.coord2radial(edge_index, coord)
        envelope = self.cutoff_envelope(radial) if self.smooth_cutoff else None

        messages = self.message_model(h[row], h[col], radial)  # compute m_{ij}
        coord = self.coord_model(coord, edge_index, coord_diff, messages, envelope)  # update x_i
        h = self.node_model(h, edge_index, messages, envelope)  # update h_i
        return h, coord


class EGNN(nn.Module):
    """EGNN model."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        radial_cutoff: float,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        coords_agg: str = "mean",
        message_agg: str = "mean",
        n_layers: int = 4,
        smooth_cutoff: bool = True,
        rebuild_edges_every_layer: bool = True,
    ):
        """EGNN model stacking multiple E_GCL layers.

        Args:
            input_size: number of node features in the input
            num_classes: number of atom types uses for the final node embedding - including the MASK class.
            message_n_hidden_dimensions: number of hidden layers of the message (edge) MLP
            message_hidden_dimensions_size: size of the hidden layers of the message (edge) MLP
            node_n_hidden_dimensions: number of hidden layers of the node update MLP
            node_hidden_dimensions_size: size of the hidden layers of the node update MLP
            coordinate_n_hidden_dimensions: number of hidden layers of the coordinate update MLP
            coordinate_hidden_dimensions_size: size of the hidden layers of the coordinate update MLP
            radial_cutoff: cutoff distance, in Angstrom, used to build the periodic edges.
            act_fn: activation function used in the MLPs. Defaults to nn.SiLU()
            residual: if True, add a skip connection in the nodes update. Defaults to True.
            attention: if True, multiply the message output by a gated value of the output. Defaults to False.
            normalize: if True, use a normalized version of the coordinates update i.e. x_i^l - x_j^l would be a unit
                vector in eq. 4 in https://arxiv.org/pdf/2102.09844. Defaults to False.
            coords_agg: Use a weighted mean (due to smooth cutoff) or sum aggregation for the coordinates update.
                Defaults to mean.
            message_agg: Use a weighted mean (due to smooth cutoff) or sum aggregation for the messages.
                Defaults to mean.
            tanh: if True, add a tanh non-linearity after the coordinates update. Defaults to False.
            n_layers: number of E_GCL layers. Defaults to 4.
            smooth_cutoff: if True, add a cosine envelope that smoothly goes to zero at radial cutoff :
                f(r) = 0.5 * (cos(pi * r / radial_cutoff) + 1). Defaults to True.
            rebuild_edges_every_layer: if True, rebuild the periodic edges from the layer-updated positions before
                every layer. If False, build them once from the input positions. Defaults to True.
        """
        super(EGNN, self).__init__()
        self.n_layers = n_layers
        self.radial_cutoff = radial_cutoff
        self.rebuild_edges_every_layer = rebuild_edges_every_layer
        self.embedding_in = nn.Linear(input_size, node_hidden_dimensions_size)
        self.graph_layers = nn.ModuleList([])
        self.node_classification_layer = nn.Linear(
            node_hidden_dimensions_size, num_classes
        )
        for _ in range(0, n_layers):
            self.graph_layers.append(
                E_GCL(
                    input_size=node_hidden_dimensions_size,
                    output_size=node_hidden_dimensions_size,
                    message_n_hidden_dimensions=message_n_hidden_dimensions,
                    message_hidden_dimensions_size=message_hidden_dimensions_size,
                    node_n_hidden_dimensions=node_n_hidden_dimensions,
                    node_hidden_dimensions_size=node_hidden_dimensions_size,
                    coordinate_n_hidden_dimensions=coordinate_n_hidden_dimensions,
                    coordinate_hidden_dimensions_size=coordinate_hidden_dimensions_size,
                    radial_cutoff=radial_cutoff,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    coords_agg=coords_agg,
                    message_agg=message_agg,
                    tanh=tanh,
                    smooth_cutoff=smooth_cutoff,
                )
            )

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
            x: node coordinates. size is number of nodes, spatial dimension
            cell_lengths: per-sample orthogonal cell side lengths, used to build the periodic
                edges. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            estimated score in an AXL namedtuple.
                coordinates: size is number of nodes, spatial dimension
                atom types: number of nodes, number of atomic species + 1 (for MASK)
                lattice: number of nodes, spatial dimension * (spatial dimension - 1) TODO
        """
        h = self.embedding_in(h)
        edges = self._build_edges(x, cell_lengths, natoms)
        for layer_index, graph_layer in enumerate(self.graph_layers):
            if self.rebuild_edges_every_layer and layer_index > 0:
                edges = self._build_edges(x, cell_lengths, natoms)
            h, x = graph_layer(h, edges, x)
        node_classification_logits = self.node_classification_layer(h)
        model_outputs = AXL(
            A=node_classification_logits,
            X=x,
            L=torch.zeros_like(x),  # TODo
        )
        return model_outputs
