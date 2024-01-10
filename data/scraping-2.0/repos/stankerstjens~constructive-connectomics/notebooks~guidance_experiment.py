import json
import random
from functools import partial
from typing import Optional, Tuple, Iterable

import igraph
import numpy as np
from tqdm import tqdm

from abianalysis.guidance import GuidanceGraph, Axon, get_euclidean_path_length, \
    get_euclidean_distance
from abianalysis.guidance.factory import correlation_landscape, \
    normalized_weight, threshold_edge_mask
from abianalysis.hierarchy import Hierarchy
from abianalysis.hierarchy.decomposition import pca_split, \
    random_split, make_balanced_hierarchy
from abianalysis.spatial import VoxelGraph, voxel_graph_from_volume
from abianalysis.volume import Volume
from pylineage.multi_lineage_simulator import MultiLineageSimulator


def _block_mean_expression(pos: np.ndarray, exp: np.ndarray, k: int) -> \
        Tuple[np.ndarray, np.ndarray]:
    n_voxels, n_genes = exp.shape
    shape = np.max(pos, 0) - np.min(pos, 0) + 1
    mat = np.zeros((*shape, n_genes))

    p = np.array(pos)
    p -= np.min(p, axis=0)
    mat[tuple(p.T)] = exp
    mat = np.add.reduceat(mat, np.arange(0, mat.shape[0], k), axis=0)
    mat = np.add.reduceat(mat, np.arange(0, mat.shape[1], k), axis=1)
    mat = np.add.reduceat(mat, np.arange(0, mat.shape[2], k), axis=2)

    mat /= k ** 3
    pos = np.array(np.nonzero(mat.sum(axis=3))).T
    exp = mat[tuple(pos.T)]

    return pos, exp


def draw_random_axon(voxel_graph: VoxelGraph, source_voxel, n_steps):
    fringe = {source_voxel}
    visited = set()
    previous = {source_voxel: None}

    for i in range(n_steps):
        voxel = random.sample(fringe, 1)[0]
        fringe.remove(voxel)
        visited.add(voxel)

        neighbors = set(voxel_graph.get_neighbors(voxel)) - visited
        for neighbor in neighbors:
            previous[neighbor] = voxel
        fringe.update(neighbors)

    visited = sorted(visited)
    idx = {v: i for i, v in enumerate(visited)}

    tree = igraph.Graph(directed=True)
    tree.add_vertices(len(visited), attributes={'voxel': visited})
    tree.add_edges([(idx[previous[voxel]], idx[voxel])
                    for voxel in visited
                    if previous[voxel] is not None])

    axon = Axon()
    axon._tree = tree
    return axon


class AxonNavigator:
    def __init__(self, hierarchy: Hierarchy,
                 landscape_threshold: float = .1,
                 gradient_threshold: float = .0):
        """

        :param hierarchy:
        :param landscape_threshold: The smallest measurable correlation
        :param gradient_threshold: The smallest measurable correlation gradient
        """
        self.hierarchy = hierarchy
        self.landscape_threshold = landscape_threshold
        self.gradient_threshold = gradient_threshold

        self.voxel_graph = None
        self._source_vertices = []

    @property
    def volume(self):
        return self.hierarchy.volume

    def simulate_axons(self, n_axons, source_voxels=None):
        """Simulate n_axons axons. If specified, the source voxels are used.
        Otherwise, random voxels are used.
        """
        if source_voxels is None:
            source_voxels = np.random.choice(
                self.volume.n_voxels,
                size=n_axons,
                replace=False
            )
        self._source_vertices = self.guidance_graph.get_leaf_vertex(
            source_voxels)

        return [self.guidance_graph.find_axon(source) for source
                in tqdm(self.sources, desc='Sampling axons')]

    def _init_voxel_graph(self):
        self.voxel_graph = voxel_graph_from_volume(self.volume)

    def _init_guidance_graph(self):
        landscape_fn = partial(correlation_landscape,
                               threshold=self.landscape_threshold)
        weight_fn = normalized_weight
        mask_fn = partial(threshold_edge_mask,
                          threshold=self.gradient_threshold)
        self.guidance_graph = GuidanceGraph.create(
            self.hierarchy,
            self.voxel_graph,
            hierarchy_to_landscape=landscape_fn,
            gradient_to_weight=weight_fn,
            edge_mask=mask_fn,
        )


class Experiment:
    """

    :param age:
    :param n_iterations:
    :param n_sources:
    :param landscape_threshold:
    :param gradient_threshold:
    :param split_method:
    :param expression:
    :param label:

    """

    def __init__(self,
                 age: str,
                 n_iterations: int,
                 n_sources: int,
                 landscape_threshold=.1,
                 gradient_threshold=.0,
                 split_method='pca',
                 genes: Optional[np.ndarray] = None,
                 expression='',
                 label: str = '',
                 noise_amount=0.):
        # Parameters
        self.age = age
        self.n_iterations = n_iterations
        self.n_sources = n_sources
        self.label = label
        self.expression = expression
        self.noise_amount = noise_amount
        self.genes = genes

        if split_method == 'pca':
            split_method = pca_split
        elif split_method == 'random':
            split_method = random_split
        else:
            raise ValueError('Invalid split method')
        self.split_method = split_method

        self.landscape_threshold = landscape_threshold
        self.gradient_threshold = gradient_threshold

        self.volume: Optional[Volume] = None
        self.hierarchy: Optional[Hierarchy] = None
        self.voxel_graph: Optional[VoxelGraph] = None
        self.guidance_graph: Optional[GuidanceGraph] = None

        self.sources = None
        self.axons = None

    def _set_model_expression(self):
        n_genes = self.volume.n_genes
        for h in self.hierarchy.descendants():
            h._expression = np.random.randn(n_genes)
            if h.parent:
                h._expression += h.parent.expression

        for h in self.hierarchy.leaves():
            self.volume.expression[h.voxel_index] = h._expression

    def _smooth_expression(self):
        exp = self.volume.expression.copy()
        g = self.voxel_graph._graph
        for v in g.vs:
            self.volume.expression[v.index] = np.mean(exp[g.neighbors(v)],
                                                      axis=0)

    def _prepare_volume(self):
        pass

    def _optional_shuffle(self):
        if self.shuffle:
            self.volume.shuffle()

    def _add_some_noise(self, amount: float = 0.) -> None:
        """
        :param amount:  Some float between 0 and 1, 0 meaning no noise and 1
            meaning only noise.
        """
        noise = np.random.randn(*self.volume.expression.shape)
        self.volume.expression = ((1 - amount) * self.volume.expression
                                  + amount * noise)
        for leaf in self.hierarchy.leaves():
            assert leaf.voxel_index is not None, \
                'Leaves should have voxel indices'
            leaf._expression = None

    @property
    def shuffle(self):
        """True if the volume expression data should be shuffled."""
        return 'shuffled' in self.expression

    @property
    def smooth(self):
        """True if the expression should be spatially smoothed.

        This could either be before establishing the hierarchy (in case the
        expression is also shuffled) or after (in case the model expression
        is applied).

        """
        return 'smooth' in self.expression

    @property
    def model(self):
        """True if the expression should be modeled after the hierarchy."""
        return 'model' in self.expression

    def _prepare_hierarchy(self):
        if self.smooth and self.shuffle:
            self._smooth_expression()

        print('Preparing hierarchy...')
        self.hierarchy = make_balanced_hierarchy(
            self.volume,
            n_iterations=self.n_iterations,
            partition_children=self.split_method
        )

        if self.model:
            self._set_model_expression()
            if self.smooth:
                self._smooth_expression()

    def _prepare_voxel_graph(self):
        print('Preparing voxel graph...')
        self.voxel_graph = voxel_graph_from_volume(self.volume)

    def _reduce_genes(self):
        self.volume.filter_genes(self.genes)
        for h in self.hierarchy.descendants():
            if h.component is not None:
                h.component = h.component[self.genes]

    def _prepare_guidance_graph(self):
        landscape_fn = partial(correlation_landscape,
                               threshold=self.landscape_threshold)
        weight_fn = normalized_weight
        mask_fn = partial(threshold_edge_mask,
                          threshold=self.gradient_threshold)
        self.guidance_graph = GuidanceGraph.create(
            self.hierarchy,
            self.voxel_graph,
            hierarchy_to_landscape=landscape_fn,
            gradient_to_weight=weight_fn,
            edge_mask=mask_fn,
        )

    def prepare(self):
        self._prepare_volume()
        self._prepare_voxel_graph()
        self._prepare_hierarchy()
        if self.noise_amount > 0:
            self._add_some_noise(self.noise_amount)
        if self.genes is not None:
            self._reduce_genes()
        self._prepare_guidance_graph()

    def snowball(self, source_voxel=None):
        if source_voxel is None:
            source_voxel = np.random.choice(self.volume.n_voxels,
                                            size=1).item()

        self.sources = []
        self.axons = []

        visited = set()
        voxels = {source_voxel}
        i = 0
        while i < self.n_sources or voxels:
            print(i, end='\r')
            if len(voxels - visited) > 0:
                voxel = np.random.choice(list(voxels), size=1).item()
                voxels.remove(voxel)
            else:
                remaining_voxels = set(range(self.volume.n_voxels)) - visited
                voxel = np.random.choice(list(remaining_voxels), size=1).item()
            visited.add(voxel)
            source_index = self.guidance_graph.get_leaf_vertex(voxel)

            axon = self.guidance_graph.find_axon(source_index)
            voxels.update(set(axon.tips) - visited)

            self.sources.append(source_index)
            self.axons.append(axon)
            i += 1
        self.n_sources = len(self.axons)

    def sample_axons(self, source_voxels=None):
        if source_voxels is None:
            source_voxels = np.random.choice(
                self.volume.n_voxels,
                size=self.n_sources,
                replace=False
            )
        else:
            self.n_sources = len(source_voxels)
        self.sources = self.guidance_graph.get_leaf_vertex(source_voxels)
        self.axons = [self.guidance_graph.find_axon(source) for source
                      in tqdm(self.sources, desc='Sampling axons')]

    def random_fake_axons(self) -> Iterable['FakeAxon']:
        for axon in self.axons:
            axon = draw_random_axon(self.voxel_graph,
                                    axon.source_voxel,
                                    len(axon.reached_voxels))
            yield axon

    def get_path_lengths(self):
        return [get_euclidean_path_length(self.volume, path)
                for axon in tqdm(self.axons, desc='Calculating path lengths')
                for path in axon.voxel_paths
                if len(path) > 1]

    def get_path_distances(self):
        return [get_euclidean_distance(self.volume, path)
                for axon in tqdm(self.axons, desc='Calculating path distances')
                for path in axon.voxel_paths
                if len(path) > 1]

    def get_reached_voxels_counts(self):
        return [len(axon.reached_voxels) for axon
                in tqdm(self.axons, desc='Counting reached voxels')]

    def save_to_json(self, file_name: str = None):
        """Save the experiment to a json file"""

        if file_name is None:
            file_name = self.label.lower().replace(' ', '_') + '.json'

        voxel_hierarchy = np.zeros(self.volume.n_voxels, dtype=np.int32)
        for h in self.hierarchy.descendants():
            if any(c.is_leaf for c in h.children):
                voxel_hierarchy[h.voxels] = h.id

        vis_data = {
            'hierarchy': self.hierarchy.to_json_dict(
                include_only=('children')),
            'voxel_hierarchy': voxel_hierarchy.tolist(),
            'volume': {
                'voxel_indices': self.volume.voxel_indices.T.tolist(),
                **self.volume.to_json_dict(
                    include_only=('voxel_size', 'age', 'anatomy', 'id', 'name')
                )
            },
            'axons': [{
                'branch_paths': a.branch_paths,
                'source_voxel': a.source_voxel.item(),
            } for a in self.axons],
        }

        with open(file_name, 'w') as fh:
            json.dump(vis_data, fh)


class DataExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_volume(self):
        print('Preparing volume...')
        self.volume = Volume.load(self.age)
        self.volume.preprocess()

        self._optional_shuffle()


class SimulatedExperiment(Experiment):
    def __init__(self, n_voxels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_voxels = n_voxels

    def _prepare_volume(self):
        print('Simulating volume...')
        k = 2
        n_genes = len(self.genes) if self.genes is not None else 100
        mls = MultiLineageSimulator(n_dims=3,
                                    n_roots=100,
                                    n_divisions=k ** 3 * self.n_voxels,
                                    n_genes=n_genes,
                                    symmetric_prob=.2)
        mls.run()

        cells = list(mls.root.leaves())
        pos = np.array([c.position.index for c in cells])
        exp = np.array([c.state.expression for c in cells])

        pos, exp = _block_mean_expression(pos, exp, k)
        exp += np.random.randn(*exp.shape) * 5

        self.volume = Volume(expression=exp,
                             voxel_indices=pos,
                             genes=[f'g{i}' for i in range(len(exp[0]))],
                             age=self.age)

        self.volume.preprocess(anatomy=None)
        self._optional_shuffle()
