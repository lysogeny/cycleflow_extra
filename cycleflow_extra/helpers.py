import multiprocessing
import numpy as np
import cycleflow as cf

def unpacking_apply_along_axis(arg):
    """Takes arguments in a tuple"""
    func1d, axis, arr, args, kwargs = arg
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Parallel variant of np.apply_along_axis

    copied from stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
    """

    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]
    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    pool.close()
    pool.join()
    return np.concatenate(individual_results)

def tophat_posterior(bounds, timepoints, mean, err):
    """Posterior probabilitity using a tophat shaped prior from bounds"""
    prior = cf.log_flat_prior(bounds["min"], bounds["max"])
    likelihood = cf.log_likelihood(timepoints, mean, err)
    return cf.log_posterior(likelihood, prior)

