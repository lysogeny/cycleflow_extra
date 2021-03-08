"""Decorators for plotting functions"""

import numpy as np
import pandas as pd

def check_array(dim_dim=1):
    """Checks the validity of array arguments for plotting functions"""
    def inner(fun):
        """Actual decorator"""
        def innerer(samples, *args, names=None, dimnames=None, **kwargs):
            """Plotting function"""
            if type(samples) == np.ndarray:
                samples = [samples]
            elif type(samples) == dict:
                if names is None:
                    names = list(samples.keys())
                samples = list(samples.values())
            n_samples = len(samples)
            n_dims = max(map(lambda x: x.shape[dim_dim], samples))
            if not names:
                names = [None for i in range(n_samples)]
            if not dimnames:
                dimnames = [f"Dim {i}" for i in range(n_dims)]
            return fun(samples, names, dimnames, *args, **kwargs)
        return innerer
    return inner

def check_dataframes(dim_dim=3):
    """Checks that the data/simulation structure is valid"""
    def inner(fun):
        """Actual decorator"""
        def innerer(simulation, data, *args, names=None, dimnames=None, **kwargs):
            """Plotting function"""
            if type(simulation) == pd.DataFrame:
                simulation = [simulation]
            elif type(simulation) == dict:
                if names is None:
                    names = list(simulation.keys())
                simulation = list(simulation.values())
            if type(data) == pd.DataFrame:
                data = [data]
            elif type(data) == dict:
                if names is None:
                    names = list(data.keys())
                data = [data[n] for n in names]
            n_samples = len(simulation)
            assert len(data) == len(names), "Mismatched data/simulation count"
            if names is None:
                names = [None for i in range(n_samples)]
            # Stuck at: need to figure out what data should look like for these.

            # I want these to probably be:
            # data: list[pd.DataFrame]
            # Rows: timepoints
            # Columns: multiindex (group, errors)



