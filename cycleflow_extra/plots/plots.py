"""Plotting functions for working with cycleflow"""

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
    assert all([d in {x[0] for x in sim.columns} for d in dims]), "Missing columns in `sim`"
    assert all([d in {x[0] for x in data.columns} for d in dims]), "Missing columns in `data`"
    for (i, dim) in enumerate(dims):
        axes[i].errorbar(data.index, data[(dim, "mean")], yerr=data[(dim, "error")], fmt='k.')
        axes[i].plot(simulation.index, simulation[(dim, "median")], color='red')
        axes[i].fill_between(simulation.index, simulation[(dim, "lower")], simulation[(dim, "upper")], color="pink")
        axes[i].set_ylabel(dim)
    return axes

def sim_summarised(simulation: pd.DataFrame, data: pd.DataFrame, **kwargs):
    """Plots the simulation from a summarised dataframe"""
    fig, p = plt.subplots(1, len(dims), squeeze=True, **kwargs)
    _sim_summarised(p, simulation, data)
    return fig

def sims_summarised(simulations: dict, data: dict, byrow=True, **kwargs):
    """Plots the simulations from a summarised dataframe"""
    samples = set(simulations.keys()) & set(data.keys())
    if byrow:
        fig, p = plt.subplots(length(samples), 3, squeeze=False)
        p = p.transpose()
    else:
        fig, p = plt.subplots(3, length(samples), squeeze=False)
    for (i, sample) in enumerate(samples):
        _sim_summarised(p[:,i], simulations[sample], data[sample], name=sample)
    return fig, p

def fraction_summarised(samples, data=None, **kwargs):
    if type(samples) == pd.DataFrame:
        # If there is only one sample, we treat it as one.
        samples = [samples]
    elif type(samples) == dict:
        names = list(samples.keys())
        samples = list(samples.values())
    if type(data) == pd.DataFrame:
        data = [data]
    elif type(data) == dict:
        data_names = list(data.keys())
        data = [data[n] for n in names]
    n_samples = len(samples)
    fig, p = plt.subplots(1, n_samples, **kwargs)
    width = 0.35
    ixs = [-width/2, +width/2, 1-width/2, 1+width/2]
    labs = ["Model S", "Data S", "Model G2M", "Data G2M"]
    for (s, sample) in enumerate(samples):
        p[s].bar([-width/2], sample.loc["median"]["S"],
                 yerr=sample.loc[["down", "up"]]["S"].to_numpy().reshape(2,1), width=width, color='pink', edgecolor='red')
        p[s].bar([+width/2], data[s].loc["S"]["mean"],
                 yerr=data[s].loc["S"]["error"], width=width, color='lightgray', edgecolor='black')
        p[s].bar([1-width/2], sample.loc["median"]["G2M"],
                 yerr=sample.loc[["down", "up"]]["G2M"].to_numpy().reshape(2,1), width=width, color='pink', edgecolor='red')
        p[s].bar([1+width/2], data[s].loc["G2"]["mean"],
                 yerr=data[s].loc["G2"]["error"], width=width, color='lightgray', edgecolor='black')
        p[s].legend(["Model", "Data"])
        p[s].set_xticks([0,1])
        p[s].set_xticklabels(["S", "G2M"])
    return fig

def model(ts, samples, data=None, **kwargs):
    pass
