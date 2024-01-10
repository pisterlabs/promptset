from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from guidance_experiment import Experiment


def distance_vs_path_length_plot(experiment: Experiment,
                                 main: float = .75,
                                 margin: float = .1,
                                 n_bins: int = 25,
                                 max_value: Optional[float] = None,
                                 colorbar=True,
                                 ) -> Figure:
    path_lengths = experiment.get_path_lengths()
    path_distances = experiment.get_path_distances()

    fig = plt.figure(figsize=(5, 5))

    main_ax = fig.add_axes([margin, margin, main, main])
    hbins = main_ax.hexbin(path_lengths,
                           path_distances,
                           gridsize=n_bins,
                           bins='log',
                           cmap='viridis',
                           mincnt=1,
                           )
    main_ax.set_xlabel('Path length (µm)')
    main_ax.set_ylabel('Straight distance (µm)')
    if max_value is None:
        max_value = np.max([path_lengths, path_distances])
    main_ax.plot([0, max_value],
                 [0, max_value],
                 color='black',
                 linestyle='dashed')

    ax = fig.add_axes([margin, main + margin, main, 1 - main - margin],
                      sharex=main_ax)
    ax.hist(path_lengths,
            histtype='step',
            bins=n_bins,
            color='k')
    ax.axis('off')
    ax.grid()
    ax.set_title(experiment.label)

    ax = fig.add_axes([main + margin, margin, 1 - main - margin, main],
                      sharey=main_ax)
    ax.hist(path_distances,
            histtype='step',
            bins=n_bins,
            color='k',
            orientation='horizontal')
    ax.axis('off')

    if colorbar:
        fig.colorbar(
            hbins,
            cax=fig.add_axes([main - .5 * margin,
                              1.5 * margin, margin / 3,
                              margin * 2]))

    return fig


def plot_path_length_comparison(experiments: List[Experiment],
                                n_bins=30, histtype='step'):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist([
        exp.get_path_lengths()
        for exp in experiments
    ],
        bins=n_bins,
        histtype=histtype,
        label=[exp.label for exp in experiments],
    )
    ax.set_xlabel('Path length')
    ax.set_ylabel('# Paths')

    fig.legend()

    return fig


def plot_reached_distance_comparison(experiments: List[Experiment],
                                     n_bins=30,
                                     histtype='step'):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist([
        exp.get_path_distances()
        for exp in experiments
    ],
        bins=n_bins,
        histtype=histtype,
        label=[exp.label for exp in experiments],
    )

    ax.set_xlabel('Reached distance')
    ax.set_ylabel('# Paths')

    fig.legend()
    return fig


def plot_reached_voxel_count_comparison(experiments: List[Experiment],
                                        n_bins=10,
                                        histtype='step') -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist([
        exp.get_reached_voxels_counts()
        for exp in experiments
    ],
        histtype=histtype,
        label=[exp.label for exp in experiments],
        bins=n_bins,
    )

    ax.set_yscale('log')

    ax.set_xlabel('# Reached voxels')
    ax.set_ylabel('# Axons')

    fig.legend()
    return fig


def plot_correlation_distribution(experiments: List[Experiment], n_bins=50):
    fig = plt.figure()
    ax = fig.add_subplot()

    values = []
    for exp in experiments:
        v = np.array(exp.guidance_graph._up_graph.vs['landscape'])
        v = v[(v > 0) & (v < 100)]
        values.append(v)

    ax.hist(values,
            histtype='step',
            label=[exp.label for exp in experiments],
            bins=n_bins)

    fig.legend()

    return fig
