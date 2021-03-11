"""Plotting functions for working with cycleflow"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import decorators

#def chains(samples, *args, dimnames=None, **kwargs):
#    """Plot the sampled chains
#
#    Takes a numpy array of shape
#    (n_samples, n_chains, n_dims) and plots all dims
#    """
#    _, n_chains, n_dims = samples.shape
#    if not dimnames:
#        dimnames = [f"Dim {i}" for i in range(n_dims)]
#    fig, p = plt.subplots(n_dims, 1, sharex=True, **kwargs)
#    for i in range(n_dims):
#        p[i].set_ylabel(dimnames[i])
#        for j in range(n_chains):
#            p[i].plot(samples[:,j,i], *args)
#    p[n_dims-1].set_xlabel("samples")
#    plt.close(fig)
#    return fig

@decorators.check_array(dim_dim=2)
def chains(samples, names, dimnames, *args, **kwargs):
    """Plot the sampled chains

    Takes a numpy array of shape
    (n_samples, n_chains, n_dims) and plots all dims
    """
    n_samples = len(samples)
    n_dims = len(dimnames)
    fig, p = plt.subplots(n_dims, n_samples, sharex=True, squeeze=False, **kwargs)
    for (s,sample) in enumerate(samples):
        n_chains = sample.shape[1]
        for i in range(n_dims):
            p[i][0].set_ylabel(dimnames[i])
            for j in range(n_chains):
                p[i][s].plot(sample[:,j,i], *args)
        p[n_dims-1][s].set_xlabel("samples")
        p[0][s].set_title(names[s])
    return fig

@decorators.check_array(dim_dim=1)
def samples(samples, names, dimnames, *args, **kwargs):
    """Plot the samples as histograms

    Takes a numpy array of shape (n_samples, n_dims)
    """
    n_samples = len(samples)
    n_dims = len(dimnames)
    fig, p = plt.subplots(n_dims, n_samples, squeeze=False, **kwargs)
    for (s, sample) in enumerate(samples):
        n_chains = sample.shape[1]
        for i in range(n_dims):
            p[i][0].set_ylabel(dimnames[i])
            p[i][s].hist(sample[:,i], *args, orientation='horizontal')
        p[n_dims-1][s].set_xlabel("samples")
        p[0][s].set_title(names[s])
    return fig


def _sim_summarised(axes, simulation, data):
    """Plots the simulation and data on specified axes (3-length np array)"""
    dims = ["g01", "s", "g2m"]
    stats_data = ["mean", "error"]
    stats_sim = ["lower", "median", "upper"]
    cols_data = itertools.product(dims, stats_data)
    cols_sim = itertools.product(dims, stats_sim)
    assert all([d in {x[0] for x in simulation.columns} for d in dims]), "Missing columns in `sim`"
    assert all([d in {x[0] for x in data.columns} for d in dims]), "Missing columns in `data`"
    for (i, dim) in enumerate(dims):
        axes[i].errorbar(data.index, data[(dim, "mean")], yerr=data[(dim, "error")], fmt='k.')
        axes[i].plot(simulation.index, simulation[(dim, "median")], color='red')
        axes[i].fill_between(simulation.index, simulation[(dim, "lower")], simulation[(dim, "upper")], color="pink")
        axes[i].set_ylabel(f"fraction {dim} EdU+")
        axes[i].set_xlabel("time")
    return axes

def sim_summarised(simulation: pd.DataFrame, data: pd.DataFrame, **kwargs):
    """Plots the simulation from a summarised dataframe"""
    dims = ["g01", "s", "g2m"]
    fig, p = plt.subplots(1, len(dims), squeeze=True, **kwargs)
    _sim_summarised(p, simulation, data)
    return fig

def sims_summarised(simulations: dict, data: dict, byrow=True, **kwargs):
    """Plots the simulations from a dict of summarised dataframes"""
    samples = set(simulations.keys()) & set(data.keys())
    if byrow:
        fig, p = plt.subplots(len(samples), 3, squeeze=False)
        p = p.transpose()
    else:
        fig, p = plt.subplots(3, len(samples), squeeze=False)
    for (i, sample) in enumerate(samples):
        _sim_summarised(p[:,i], simulations[sample], data[sample])
        p[1,i].set_title(sample)
    return fig, p

def _fraction_summarised(axes, sim, data):
    """Plots the simulation and data on the axes objects specified"""
    dims = ['s', 'g2m']
    stats_data = ['mean', 'error']
    stats_sim = ['lower', 'median', 'upper']
    assert all([d in sim.index for d in dims]), "Missing indices in `sim`"
    assert all([d in sim.columns for d in stats_sim]), "Missing columns in `sim`"
    assert all([d in data.index for d in dims]), "Missing indices in `data`"
    assert all([d in data.columns for d in stats_data]), "Missing columns in `data`"
    for i, d in enumerate(dims):
        simerror = np.abs(np.array(sim.loc[d][["lower", "upper"]]).reshape((2,1)) - sim.loc[d]["median"])
        axes[i].bar("data", data.loc[d]["mean"], yerr=data.loc[d]["error"], fc='lightgray', ec='black', ecolor='black')
        axes[i].bar("model", sim.loc[d]["median"], yerr=simerror, fc='pink', ec='red', ecolor='pink')
        axes[i].set_ylabel(f"{d} in steady state")

def fraction_summarised(sim, data, **kwargs):
    dims = ['s', 'g2m']
    fig, p = plt.subplots(1, len(dims), squeeze=True, **kwargs)
    _fraction_summarised(p, sim, data)
    return fig

def fractions_summarised(sim, data, byrow=True, **kwargs):
    """Plots the steady state fractions"""
    samples = set(sim.keys()) & set(data.keys())
    if byrow:
        fig, p = plt.subplots(len(samples), 2, squeeze=False)
        p = p.transpose()
    else:
        fig, p = plt.subplots(2, len(samples), squeeze=False)
    for (i, sample) in enumerate(samples):
        _fraction_summarised(p[:,i], sim[sample], data[sample])
        p[0,i].set_title(sample)
    return fig, p

def combined_summarised(edu_sim, edu_data, ss_sim, ss_data, **kwargs):
    """Plot combined fit"""
    fig = plt.figure(**kwargs)
    axd = fig.subplot_mosaic(
        [["edu_1.g01", "edu_2.s", "edu_3.g2m", "ss_1.s", "ss_2.g2m"]],
        gridspec_kw={"width_ratios": (0.25, 0.25, 0.25, 0.125, 0.125)}
    )
    edu_panes = [v for (k, v) in axd.items() if 'edu' in k]
    ss_panes = [v for (k, v) in axd.items() if 'ss' in k]
    _sim_summarised(edu_panes, edu_sim, edu_data)
    _fraction_summarised(ss_panes, ss_sim, ss_data)
    return fig

def combineds_summarised(edu_sim, edu_data, ss_sim, ss_data, **kwargs):
    """Plot combined fit"""
    samples = edu_sim.keys() & edu_data.keys() & ss_sim.keys() & ss_data.keys()
    fig = plt.figure(**kwargs)
    axd = fig.subplot_mosaic(
        [[f"{s}::edu_1.g01",
          f"{s}::edu_2.s",
          f"{s}::edu_3.g2m",
          f"{s}::ss_1.s",
          f"{s}::ss_1.g2m"]
         for s in samples
         ],
        gridspec_kw={"width_ratios": (0.25, 0.25, 0.25, 0.125, 0.125)}
    )
    edu_panes = {k: v for (k, v) in axd.items() if 'edu' in k}
    ss_panes = {k: v for (k, v) in axd.items() if 'ss' in k}
    for s in samples:
        axd[f"{s}::edu_1.g01"].set_title(s)
        e_panes = [v for (k, v) in sorted(edu_panes.items()) if k.startswith(f"{s}::")]
        s_panes = [v for (k, v) in sorted(ss_panes.items()) if k.startswith(f"{s}::")]
        _sim_summarised(e_panes, edu_sim[s], edu_data[s])
        _fraction_summarised(s_panes, ss_sim[s], ss_data[s])
    return fig



def model(ts, samples, data=None, **kwargs):
    pass
