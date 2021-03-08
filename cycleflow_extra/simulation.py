"""Functions for forward simulations"""
import cycleflow as cf
import numpy as np
import scipy.integrate

def steady_growth(theta):
    """Find the steady growth distribution and kappa for `theta`"""
    l = abs(int(theta[4])) #number of substeps in G1
    m = 15 # number of substeps in G2M
    n = 15 # number of substeps in S
    a = theta[5] # probability to enter G0 upon mitosis
    earlyS = int(theta[6] * n)
    lateS = int(theta[7] * n)
    y0 = np.zeros(l+n+m+1)
    # construct the transition matrix
    # calculate the steady-growth state
    transitions = cf.cycleflow._make_transitions(theta)
    eig = np.linalg.eig(transitions)
    index = np.argmax(eig[0])
    k = eig[0][index]
    k = k.real
    ss_fractions = np.ascontiguousarray(eig[1][:, index].real)
    ss_fractions /= np.sum(ss_fractions)
    return (k, ss_fractions)

def steady_growth_ratio(theta):
    """Find the steady growth distribution for `theta`"""
    return steady_growth(theta)[1]

def forward_labels(theta, ts):
    """Forward simulation for the model's labelling dynamics

    Parameters:
        - theta: parameters
        - ts: list of times to evaluate at

    This function is copied and modified from cycleflow's likelihood function.
    """
    # Establish parameters
    l = abs(int(theta[4])) #number of substeps in G1
    m = 15 # number of substeps in G2M
    n = 15 # number of substeps in S
    a = theta[5] # probability to enter G0 upon mitosis
    earlyS = int(theta[6] * n)
    lateS = int(theta[7] * n)
    #
    y0 = np.zeros(l+n+m+1)
    # construct the transition matrix
    k, ss_fractions = steady_growth(theta)
    # calculate the steady-growth state
    ss_G1, ss_S, ss_G2, ss_G0 = np.split(ss_fractions, [l, l+m, l+m+n])
    ss_earlyS, ss_midS, ss_lateS = np.split(ss_S, [earlyS, -lateS])
    ss_gate_S = np.sum(ss_midS)
    ss_gate_G2 = np.sum(ss_lateS) + np.sum(ss_G2)
    labeling = np.zeros((l+n+m+1, l+m+n+1)) # allocate labeling matrix for speed
    sol = scipy.integrate.solve_ivp(cf.model_jit, [0, np.max(ts)], y0, t_eval=ts,
                                    args=(cf.cycleflow._make_transitions(theta), theta, ss_fractions, k, labeling)).y
    fit_G1l = np.sum(sol[0:l+earlyS, :], axis=0)
    fit_G0l = sol[l+m+n, :]
    fit_G0G1l = fit_G1l + fit_G0l
    fit_Sl = np.sum(sol[l+earlyS:l+n-lateS, :], axis=0)
    fit_G2l = np.sum(sol[l+n-lateS:l+n+m, :], axis=0)
    return np.vstack((fit_G0G1l, fit_Sl, fit_G2l))

def forward_ratio(theta):
    """Forward simulation for the model's gate ratios

    Parameters:
        - theta: parameters

    This function is copied and modified from cycleflow's likelihood function.
    """
    # Establish parameters
    l = abs(int(theta[4]))
    m = n = 15
    earlyS = int(theta[6] * n)
    lateS = int(theta[7] * n)
    # Compute steady state ratio
    ratios = steady_growth_ratio(theta)
    # Munge the values
    _, ss_S, ss_G2, _ = np.split(ratios, [l, l+m, l+m+n])
    _, ss_midS, ss_lateS = np.split(ss_S, [earlyS, -lateS])
    ss_gate_S = np.sum(ss_midS)
    ss_gate_G2 = np.sum(ss_lateS) + np.sum(ss_G2)
    return np.vstack((ss_gate_S, ss_gate_G2))
