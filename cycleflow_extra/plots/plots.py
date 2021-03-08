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

#@decorators.check_dataframes()
def sims_summarised(ts, samples, names=None, data_times=None, data=None, dimnames=["G01", "S", "G2M"], **kwargs):
    """Plots the simulations from a summarisde array

    Takes a vector with times and a numpy array of shape (3,3,t)
    (quantiles, dims, times)
    """
    # TODO: make this function less messy
    # TODO: catch more potential edge cases or reduce type support?
    if type(samples) == np.ndarray:
        # If there is only one sample, we treat it as one.
        samples = [samples]
    elif type(samples) == dict:
        names = list(samples.keys())
        samples = list(samples.values())
    if type(data) == np.ndarray:
        data = [data]
        data_times = [data_times]
    elif type(data) == dict:
        data_names = list(data.keys())
        data_times = [data_times[n] for n in data_names]
        data = [data[n] for n in names]
    n_samples = len(samples)
    assert len(data) == n_samples, "Dimension mismatch between times and data"
    if not names:
        names = [None for i in range(n_samples)]
    n_dims = max(map(lambda x: x.shape[1], samples))
    n_times = max(map(lambda x: x.shape[2], samples))
    assert n_times == len(ts), "Dimension mismatch between times and data"
    data_n_dims = max(map(lambda x: x.shape[1], data))
    data_n_times = max(map(lambda x: x.shape[2], data))
    assert data_n_dims == n_dims, "Dimension mismatch between data and simulation"
    if not dimnames:
        dimnames = [f"Dim {i}" for i in range(n_dims)]
    fig, p = plt.subplots(n_dims, n_samples, squeeze=False, **kwargs)
    for (s, sample) in enumerate(samples):
        d = data[s]
        for i in range(n_dims):
            p[i][s].plot(ts, sample[1,i,:], color='red')
            p[i][s].fill_between(ts, sample[0,i,:], sample[2,i,:], color='pink')
            p[i][s].errorbar(data_times[s], d[0,i,:], yerr=d[1,i,:],
                             fmt='.k')
        p[0][s].set_title(names[s])
        p[n_dims-1][s].set_xlabel("Time")
    for i in range(n_dims):
        p[i][0].set_ylabel(dimnames[i])
    return fig

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
