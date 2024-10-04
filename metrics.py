import ot
import numpy as np
import pandas as pd

import torch
from torch.optim import Adam


def mean_loglik(n,prefflow,minibatch,dataset,ranking):
    prefflow.eval()
    if n <= 5000: #too many points leads to out of RAM
        mean_loglikelihood = prefflow.logposterior(minibatch(dataset,batch_size=n,ranking=ranking),0.0).to('cpu').detach() / n
    else:
        mean_loglikelihood = 0
        bsize = min(1000,n)
        for i in range(100):
            mean_loglikelihood += prefflow.logposterior(minibatch(dataset,batch_size=bsize,ranking=ranking),0.0).to('cpu').detach()
        mean_loglikelihood /= (100*bsize)
    prefflow.train()
    return float(mean_loglikelihood.numpy())


def wasserstein_dist(flowsample,targetsample):
        if isinstance(flowsample, torch.Tensor):
            flowsample = flowsample.numpy()
        if isinstance(targetsample, torch.Tensor):
            targetsample = targetsample.numpy()
        M = ot.dist(flowsample,targetsample, metric='euclidean')
        Wd = ot.emd2([], [], M) # Compute the Wasserstein distance
        return Wd

def statistics(flowsample,variable_names):

    if isinstance(flowsample, torch.Tensor):
        flowsample = flowsample.numpy()

    data = flowsample
    # Descriptive Statistics
    print(data)
    print(data.shape)
    mean = np.mean(data, axis=0) #checkt this TypeError: mean() received an invalid combination of arguments - got (out=NoneType, dtype=NoneType, axis=int, ), but expected one of:
    std_dev = np.std(data, axis=0)
    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    percentiles = np.percentile(data, [25, 50, 75], axis=0)
    stats = np.vstack([
        mean,
        std_dev,
        minimum,
        percentiles[0],  # 25th percentile
        percentiles[1],  # 50th percentile (median)
        percentiles[2],  # 75th percentile
        maximum
    ])
    # Set option to display all columns (None means no limit)
    pd.set_option('display.max_columns', None)
    # Set option to display all rows (None means no limit)
    pd.set_option('display.max_rows', None)
    statistics = pd.DataFrame(stats, columns=variable_names, index=['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    return statistics




#total variation distance

from typing import Optional, Union

from scipy import fftpack
from scipy.optimize import brentq
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d


def _linear_binning(samples: np.ndarray, grid_points: np.ndarray):
    """Fast computation of histogram counts using a linearly spaced grid.

    Parameters
    ----------
    samples : np.ndarray
        The samples to be binned.
    grid_points : np.ndarray
        The grid points represent the bin centers. The grid points need to be
        linearly spaced (no check is performed to ensure that).

    Returns
    -------
    counts : np.ndarray
        Number of samples in each bin.
    """
    samples = samples[
        np.logical_and(samples >= grid_points[0], samples <= grid_points[-1])
    ]
    dx = grid_points[1] - grid_points[0]
    idx = np.floor((samples - (grid_points[0] - 0.5 * dx)) / dx)
    u, u_counts = np.unique(idx, return_counts=True)
    counts = np.zeros(len(grid_points))
    counts[u.astype(int)] = u_counts

    return counts


def _fixed_point(t: float, N: int, irange_squared: np.ndarray, a2: np.ndarray):
    irange_squared = np.asfarray(irange_squared, dtype=np.float64)
    a2 = np.asfarray(a2, dtype=np.float64)
    ell = 7
    f = (
        2.0
        * np.pi ** (2 * ell)
        * np.sum(
            np.power(irange_squared, ell)
            * a2
            * np.exp(-irange_squared * np.pi**2.0 * t)
        )
    )

    if f <= 0:
        return -1

    for s in reversed(range(2, ell)):
        odd_numbers_prod = np.product(
            np.arange(1, 2 * s + 1, 2, dtype=np.float64)
        )
        K0 = odd_numbers_prod / np.sqrt(2.0 * np.pi)
        const = (1.0 + (1.0 / 2.0) ** (s + 1.0 / 2.0)) / 3.0
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))
        f = (
            2.0
            * np.pi ** (2.0 * s)
            * np.sum(
                np.power(irange_squared, s)
                * a2
                * np.exp(-irange_squared * np.pi**2.0 * time)
            )
        )

    t_opt = np.power(2.0 * N * np.sqrt(np.pi) * f, -2.0 / 5.0)

    return t - t_opt


def _root(function: callable, N: int, args: tuple):
    """Try to find the smallest root whenever there is more than one.

    Root finding algorithm based on the MATLAB code by Botev et al. (2010).
    """
    N = max(min(1050.0, N), 50.0)
    tol = 1e-12 + 0.01 * (N - 50.0) / 1000.0
    converged = False
    while not converged:
        try:
            x, res = brentq(
                function, 0, tol, args=args, full_output=True, disp=False
            )
            converged = bool(res.converged)
        except ValueError:
            x = 0.0
            tol *= 2.0
            converged = False
        if x <= 0.0:
            converged = False
        if tol >= 1:
            return None

    if x <= 0.0:
        return None
    return x


def _scottrule1d(samples: np.ndarray):
    """Compute the kernel bandwidth according to Scott's rule for 1D samples.

    Parameters
    ----------
    samples : np.ndarray
        The 1D samples for which Scott's rule is being computed.

    Returns
    -------
    bandwidth : float
        Scott's bandwidth.
    """
    sigma = np.std(samples, ddof=1)
    sigma_iqr = (
        np.quantile(samples, q=0.75) - np.quantile(samples, q=0.25)
    ) / 1.3489795003921634
    sigma = min(sigma, sigma_iqr)
    return sigma * np.power(len(samples), -1.0 / 5.0)


def _validate_kde1d_args(n, lower_bound, upper_bound):
    """
    _validate_kde1d_args and raise value exception
    """
    if n <= 0:
        raise ValueError("n cannot be <= 0")

    if lower_bound is not None and upper_bound is not None:
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot be > upper_bound")

def _validate_kde1d_args(n, lower_bound, upper_bound):
    """
    _validate_kde1d_args and raise value exception
    """
    if n <= 0:
        raise ValueError("n cannot be <= 0")

    if lower_bound is not None and upper_bound is not None:
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot be > upper_bound")

def kde1d(
    samples: np.ndarray,
    n: int = 2**14,
    lower_bound: float = None,
    upper_bound: float = None,
):
    r"""Reliable and extremely fast kernel density estimator for 1D data.

    One-dimensional kernel density estimator based on fast Fourier transform.
    A Gaussian kernel is assumed and the bandwidth is chosen automatically
    using the technique developed by Botev et al. (2010) [1]_.

    Parameters
    ----------
    samples : np.ndarray
        The samples from which the density estimate is computed.
    n : int, optional
        The number of mesh points used in the uniform discretization of the
        interval [lower_bound, upper_bound]; n has to be a power of two;
        if n is not a power of two, it is rounded up to the next power of two,
        i.e., n is set to n=2^ceil(log2(n)), by default 2**14.
    lower_bound : float, optional
        The lower bound of the interval in which the density is being computed,
        if not given the default value is lower_bound=min(samples)-range/10,
        where range=max(samples)-min(samples), by default None.
    upper_bound : float, optional
        The upper bound of the interval in which the density is being computed,
        if not given the default value is upper_bound=max(data)+Range/10,
        where range=max(samples)-min(samples), by default None.

    Returns
    -------
    density : np.ndarray
        1D vector of length n with the values of the kernel density estimate
        at the grid points.
    xmesh : np.ndarray
        1D vector of grid over which the density estimate is computed.
    bandwidth : np.ndarray
        The optimal bandwidth (Gaussian kernel assumed).

    Notes
    -----
    This implementation is based on the MATLAB implementation by Zdravko Botev,
    and was further inspired by the Python implementations by Daniel B. Smith
    and the bandwidth selection code in KDEpy [2]_. We thank Zdravko Botev for
    useful clarifications on the implementation of the fixed_point function.

    Unlike other implementations, this one is immune to problems caused by
    multimodal densities with widely separated modes (see example). The
    bandwidth estimation does not deteriorate for multimodal densities because
    a parametric model is never assumed for the data.

    References
    ----------
    .. [1] Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
       estimation via diffusion. The Annals of Statistics,
       38(5):2916-2957, 2010.
    .. [2] https://github.com/tommyod/KDEpy/blob/master/KDEpy/bw_selection.py

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from numpy.random import randn

        samples = np.concatenate(
            (randn(100, 1), randn(100, 1) * 2 + 35, randn(100, 1) + 55)
        )
        kde1d(samples, 2 ** 14, min(samples) - 5, max(samples) + 5)

    """
    samples = samples.ravel()  # make samples a 1D array

    # validate values passed to the function
    _validate_kde1d_args(n, lower_bound, upper_bound)

    n = int(2 ** np.ceil(np.log2(n)))  # round up to the next power of 2
    if lower_bound is None or upper_bound is None:
        minimum = np.min(samples)
        maximum = np.max(samples)
        delta = maximum - minimum
        if lower_bound is None:
            lower_bound = np.array([minimum - 0.1 * delta])
        if upper_bound is None:
            upper_bound = np.array([maximum + 0.1 * delta])

    delta = upper_bound - lower_bound
    xmesh = np.linspace(lower_bound, upper_bound, n)
    N = len(np.unique(samples))

    initial_data = _linear_binning(samples, xmesh)
    initial_data = initial_data / np.sum(initial_data)

    # Compute the Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data, type=2)

    # Compute the bandwidth
    irange_squared = np.arange(1, n, dtype=np.float64) ** 2.0
    a2 = a[1:] ** 2.0 / 4.0
    t_star = _root(_fixed_point, N, args=(N, irange_squared, a2))

    if t_star is None:
        # Automated bandwidth selection failed, use Scott's rule
        bandwidth = _scottrule1d(samples)
        t_star = (bandwidth / delta) ** 2.0
    else:
        bandwidth = np.sqrt(t_star) * delta

    # Smooth the discrete cosine transform of initial data using t_star
    a_t = a * np.exp(
        -np.arange(n, dtype=float) ** 2 * np.pi**2.0 * t_star / 2.0
    )

    # Diving by 2 because of the implementation of fftpack.idct
    density = fftpack.idct(a_t) / (2.0 * delta)
    density[density < 0] = 0.0  # remove negatives due to round-off error

    return density.ravel(), xmesh.ravel(), bandwidth

def mtv(
    X1: Optional[Union[np.ndarray, callable]] = None,
    X2: Optional[Union[np.ndarray, callable]] = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[np.ndarray, Exception]:
    """
    Marginal total variation distances between two sets of posterior samples.

    Compute the total variation distance between posterior samples X1 and
    posterior samples X2, separately for each dimension (hence
    "marginal" total variation distance, MTV).

    Parameters
    ----------
    X1 : np.ndarray or callable, optional
        A ``N1``-by-``D`` matrix of samples, typically N1 = 1e5.
        Alternatively, may be a callable ``X1(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    X2 : np.ndarray or callable, optional
        Another ``N2``-by-``D`` matrix of samples, typically N2 = 1e5.
        Alternatively, may be a callable ``X2(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.

    Returns
    -------
    mtv: np.ndarray
        A ``D``-element vector whose elements are the total variation distance
        between the marginal distributions of ``vp`` and ``vp1`` or ``samples``,
        for each coordinate dimension.

    Raises
    ------
    ValueError
        Raised if neither ``vp2`` nor ``samples`` are specified.

    Notes
    -----
    The total variation distance between two densities `p1` and `p2` is:

    .. math:: TV(p1, p2) = \\frac{1}{2} \\int | p1(x) - p2(x) | dx.

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples/callable or posterior provided.")
    if posterior is not None:
        try:  # Get analytical marginals, if possible
            X1, bounds_1 = posterior.get_marginals()
        except AttributeError:  # Otherwise use samples
            X1 = posterior.get_samples()
            if isinstance(X1, Exception):
                return X1  # Record errors, if any
        try:  # Get analytical marginals, if possible
            X2, bounds_2 = posterior.task.get_marginals()
        except AttributeError:  # Otherwise use samples
            X2 = posterior.task.get_posterior_samples()
            if isinstance(X2, Exception):
                return X2  # Record errors, if any
        D = posterior.task.D
    else:
        D = X1.shape[1]

    nkde = 2**13
    mtv = np.zeros((D,))

    # Compute marginal total variation
    for d in range(D):

        if not callable(X1):
            yy1, x1mesh, _ = kde1d(X1[:, d], nkde)
            # Ensure normalization
            yy1 = yy1 / simpson(yy1, x1mesh)

            def f1(x):
                return interp1d(
                    x1mesh,
                    yy1,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f1(x):
                return X1(x, d).ravel()  # Analytical marginal

            x1mesh = bounds_1[:, d]  # Marginal bounds

        if not callable(X2):
            yy2, x2mesh, _ = kde1d(X2[:, d], nkde)
            # Ensure normalization
            yy2 = yy2 / simpson(yy2, x2mesh)

            def f2(x):
                return interp1d(
                    x2mesh,
                    yy2,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f2(x):
                return X2(x, d).ravel()  # Analytical marginal

            x2mesh = bounds_2[:, d]  # Marginal bounds

        def f(x):
            return np.abs(f1(x) - f2(x))

        lb = min(x1mesh[0], x2mesh[0])
        ub = max(x1mesh[-1], x2mesh[-1])
        if not np.isinf(lb) and not np.isinf(ub):
            # Grid integration (faster)
            grid = np.linspace(lb, ub, int(1e6))
            y_tot = f(grid)
            mtv[d] = 0.5 * simpson(y_tot, grid)
        else:
            # QUADPACK integration (slower)
            mtv[d] = 0.5 * quad(f, lb, ub)[0]
    return mtv


def mmtv(
    X1 = None,
    X2 = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[float, Exception]:
    """
    Mean marginal total variation dist. between two set of posterior samples.
    """
    if isinstance(X1, torch.Tensor):
        X1 = X1.numpy()
    if isinstance(X2, torch.Tensor):
        X2 = X2.numpy()
    result = mtv(X1, X2, posterior)
    if isinstance(result, Exception):
        return result
    else:
        return result.mean()