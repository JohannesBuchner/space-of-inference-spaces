import numpy as np
from numpy import pi, sin, log, cos

## speed-ups for loggamma:
import scipy.special
import functools
origgammaln = functools.lru_cache(scipy.special.gammaln)
def mygammaln(c):
    if np.shape(c) == ():
        return origgammaln(c)
    o = c.copy()
    o[:] = origgammaln(c[0])
    return o
scipy.special.gammaln = mygammaln
## end speed-up hack

import scipy.stats

def autoparamnames(ndim):
    """Generate names for parameters."""
    return ['param%d' % (i+1) for i in range(ndim)]

def null_transform(x):
    return np.copy(x)


def get_asymgauss(ndim):
    """d-dimensional Gaussian with diverse standard deviations."""
    sigmamin = 10**(-10 + ndim**0.5/2)
    if ndim == 4:
        assert np.isclose(sigmamin, 1e-9), sigmamin
    if ndim == 16:
        assert np.isclose(sigmamin, 1e-8), sigmamin
    if ndim == 100:
        assert np.isclose(sigmamin, 1e-5), sigmamin
    sigma = np.logspace(-1, np.log10(sigmamin), ndim)
    width = 1 - 5 * sigma
    width[width < 1e-20] = 1e-20
    centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2).sum()
        like[~np.isfinite(like)] = -1e300
        return like

    return autoparamnames(ndim), loglike, null_transform, True, (centers, sigma)

def get_stdfunnel(ndim, gamma=0):
    """Neal's funnel problem

    gamma controls the correlation.

    uses lnvar as first parameter

    """
    assert ndim > 0
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    prefactor = log(2 * pi) * ndim

    paramnames = ['lnvar'] + ['p%d' % i for i in range(ndim)]

    if gamma == 0:
        def loglike(theta):
            sigma = np.exp(theta[0] * 0.5)
            like = -0.5 * (((theta[1:])/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
        def transform(x):
            z = x * 200 - 100
            z[0] = scipy.stats.norm.ppf(x[0])
            return z
        return paramnames, loglike, transform, False, None

    def loglike_vectorized(theta):
        var = np.exp(theta[:,0])
        r = np.einsum('ij,jk,ik->i', theta[:,1:], Minv, theta[:,1:]) / var
        like = -0.5 * (prefactor + logMdet + theta[:,0] * ndim + r)
        return np.where(like > -1e300, like, -1e300)
    def transform_vectorized(x):
        z = x * 200 - 100
        z[:,0] = scipy.stats.norm.ppf(x[:,0])
        return z

    return paramnames, loglike_vectorized, transform_vectorized, True, None


def get_stdfunnel2(ndim, gamma=0):
    """Neal's funnel problem

    gamma controls the correlation.

    uses var as first parameter

    """
    assert ndim > 0
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    prefactor = log(2 * pi) * ndim

    paramnames = ['var'] + ['p%d' % i for i in range(ndim)]

    if gamma == 0:
        def loglike(theta):
            sigma = theta[0]
            like = -0.5 * (((theta[1:])/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
        def transform(x):
            z = x * 200 - 100
            z[0] = np.exp(scipy.stats.norm.ppf(x[0]) * 0.5)
            return z
        return paramnames, loglike, transform, False, None

    def loglike_vectorized(theta):
        var = theta[:,0]
        r = np.einsum('ij,jk,ik->i', theta[:,1:], Minv, theta[:,1:]) / var
        like = -0.5 * (prefactor + logMdet + theta[:,0] * ndim + r)
        return np.where(like > -1e300, like, -1e300)
    def transform_vectorized(x):
        z = x * 200 - 100
        z[:,0] = np.exp(scipy.stats.norm.ppf(x[:,0]) * 0.5)
        return z

    return paramnames, loglike_vectorized, transform_vectorized, True, None


def get_stdfunnel3(ndim, gamma=0):
    """Neal's funnel problem

    gamma controls the correlation.

    uses sigma as first parameter, and scaled deviation from mean as the others.

    """
    assert ndim > 0
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    prefactor = log(2 * pi) * ndim

    paramnames = ['sigma'] + ['dp%d' % i for i in range(ndim)]

    if gamma == 0:
        def loglike(theta):
            sigma = theta[0]
            x = theta[1:] * sigma
            like = -0.5 * (((x)/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
        def transform(x):
            z = x * 200 - 100
            z[0] = np.exp(scipy.stats.norm.ppf(x[0]) * 0.5)
            z[1:] /= z[0]
            return z
        return paramnames, loglike, transform, False, None

    def loglike_vectorized(theta):
        sigma = theta[:,0]
        var = sigma**2
        x = theta[:,1:] * sigma.reshape((-1, 1))
        r = np.einsum('ij,jk,ik->i', x, Minv, x) / var
        like = -0.5 * (prefactor + logMdet + sigma * ndim + r)
        return np.where(like > -1e300, like, -1e300)
    def transform_vectorized(x):
        z = x * 200 - 100
        z[:,0] = np.exp(scipy.stats.norm.ppf(x[:,0]) * 0.5)
        z[:,1:] /= z[:,0].reshape((-1, 1))
        return z

    return paramnames, loglike_vectorized, transform_vectorized, True, None


def get_stdfunnel4(ndim, gamma=0):
    """Neal's funnel problem

    gamma controls the correlation.

    uses sigma as first parameter, and scaled deviation from mean as the others.

    """
    assert ndim > 0
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    prefactor = log(2 * pi) * ndim

    paramnames = ['sigma'] + ['dp%d' % i for i in range(ndim)]

    if gamma == 0:
        def loglike(theta):
            sigma = theta[0]
            x = theta[1:] * sigma
            like = -0.5 * (((x)/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
        def transform(x):
            z = x * 20 - 10
            z[0] = np.exp(scipy.stats.norm.ppf(x[0]) * 0.5)
            z[1:] /= z[0]
            return z
        return paramnames, loglike, transform, False, None

    def loglike_vectorized(theta):
        sigma = theta[:,0]
        var = sigma**2
        x = theta[:,1:] * sigma.reshape((-1, 1))
        r = np.einsum('ij,jk,ik->i', x, Minv, x) / var
        like = -0.5 * (prefactor + logMdet + sigma * ndim + r)
        return np.where(like > -1e300, like, -1e300)
    def transform_vectorized(x):
        z = x * 20 - 10
        z[:,0] = np.exp(scipy.stats.norm.ppf(x[:,0]) * 0.5)
        z[:,1:] /= z[:,0].reshape((-1, 1))
        return z

    return paramnames, loglike_vectorized, transform_vectorized, True, None

def get_rosenbrock(ndim):
    """Rosenbock's function problem, with -2 factor"""
    
    def loglike(theta):
        a = theta[:,:-1]
        b = theta[:,1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 20 - 10
    
    return autoparamnames(ndim), loglike, transform, True, None

def get_halfrosenbrock(ndim):
    """Rosenbock's function problem, with -1 factor"""
    
    def loglike(theta):
        a = theta[:,:-1]
        b = theta[:,1:]
        return -(100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 10 - 5
    
    return autoparamnames(ndim), loglike, transform, True, None

def get_eggbox(ndim):
    """Eggbox problem"""
    assert ndim == 2
    
    def loglike(z):
        chi = (cos(z / 2.)).prod(axis=1)
        return (2. + chi)**5

    def transform(x):
        return x * 10 * pi

    return ["a", "b"], loglike, transform, True, None

def get_beta(ndim):
    """Factorized beta distribution with random parameters."""
    rng = np.random.RandomState(ndim)
    a_values = 10**rng.uniform(-1, 1, size=ndim)
    b_values = 10**rng.uniform(-1, 1, size=ndim)
    def loglike(theta):
        logprob = scipy.stats.beta.logpdf(theta, a=a_values, b=b_values)
        logprob[~(logprob > -1e300)] = -1e300
        logprob = logprob.sum(axis=1)
        logprob[~np.isfinite(logprob)] = -1e300
        return logprob

    return autoparamnames(ndim), loglike, null_transform, True, (a_values, b_values)

def get_loggamma(ndim):
    """4-mode LogGamma problem, factorized with gaussians and loggamma distributions."""
    assert ndim >= 2
    rv1a = scipy.stats.loggamma(1, loc=2./3, scale=1./30)
    rv1b = scipy.stats.loggamma(1, loc=1./3, scale=1./30)
    rv2a = scipy.stats.norm(2./3, 1./30)
    rv2b = scipy.stats.norm(1./3, 1./30)
    rv_rest = []
    for i in range(2, ndim):
        if i <= (ndim+2)/2:
            rv = scipy.stats.loggamma(1, loc=2./3., scale=1./30)
        else:
            rv = scipy.stats.norm(2./3, 1./30)
        rv_rest.append(rv)
        del rv

    def loglike(theta):
        L1 = log(0.5 * rv1a.pdf(theta[:,0]) + 0.5 * rv1b.pdf(theta[:,0]) + 1e-300)
        L2 = log(0.5 * rv2a.pdf(theta[:,1]) + 0.5 * rv2b.pdf(theta[:,1]) + 1e-300)
        Lrest = np.sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[:,2:].transpose())], axis=0)
        like = L1 + L2 + Lrest
        like = np.where(like < -1e300, -1e300 - ((np.asarray(theta) - 0.5)**2).sum(), like)
        return like

    return autoparamnames(ndim), loglike, null_transform, True, ((rv1a, rv1b), (rv2a, rv2b), rv_rest)

def get_multisine_nonvectorized(ncomponents, Ndata = 40, contrast = 100):
    """Fit of a sinosoidal light curve with random measurement times."""
    rng = np.random.RandomState(2)
    jitter_true = 0.1
    phase_true = 0.
    period_true = 180
    amplitude_true = contrast / Ndata * jitter_true
    amplitude2_true = amplitude_true / 5
    phase2_true = 1.0
    period2_true = 44.
    offset_true = -5
    
    paramnames = {
        0 : ['offset', 'jitter'],
        1 : ['offset', 'jitter', 'amplitude', 'phase', 'period'],
        2 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
        3 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
    }[ncomponents]
    
    modelx = np.linspace(0, 360, 1000)
    modely = offset_true + amplitude_true * sin(modelx / period_true * 2 * pi + phase_true) + amplitude2_true * sin(modelx / period2_true * 2 * pi + phase2_true)
    x = rng.uniform(0, 360, Ndata)
    y = rng.normal(offset_true + amplitude_true * sin(x / period_true * 2 * pi + phase_true) + amplitude2_true * sin(x / period2_true * 2 * pi + phase2_true), jitter_true)
    data = dict(
        modelx = modelx,
        modely = modely,
        datax = x,
        datay = y,
        yerr = jitter_true,
    )
    
    def loglike(params):
        offset, jitter = params[:2]
        
        amplitudes, phases, periods = params[2::3], params[3::3], params[4::3]
        predicty = offset
        for amplitude, phase, period in zip(amplitudes, phases, periods):
            predicty = predicty + amplitude * sin(x / period * 2 * pi + phase)
        
        # gaussian centered at "predicty" and uncertainties "jitter"
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y) / jitter)**2).sum()
        return logl
    
    def transform(x):
        z = x.copy()
        z[0] = x[0] * 100 - 50
        z[1] = 10**(x[1] * 1 - 1.5)
        if len(x) > 2:
            z[2::3] = 10**(x[2::3] * 4 - 2)
            z[3::3] = 2 * pi * x[3::3]
            z[4::3] = 10**(x[4::3] * 4 - 1)
        return z
    
    return paramnames, loglike, null_transform, False, data

def get_multisine(ncomponents, Ndata = 40, contrast = 100):
    """Fit of a sinosoidal light curve with random measurement times."""
    rng = np.random.RandomState(2)
    jitter_true = 0.1
    phase_true = 0.
    period_true = 180
    amplitude_true = contrast / Ndata * jitter_true
    amplitude2_true = amplitude_true / 5
    phase2_true = 1.0
    period2_true = 44.
    offset_true = -5
    
    paramnames = {
        0 : ['offset', 'jitter'],
        1 : ['offset', 'jitter', 'amplitude', 'phase', 'period'],
        2 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
        3 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
    }[ncomponents]
    
    modelx = np.linspace(0, 360, 1000)
    modely = offset_true + amplitude_true * sin(modelx / period_true * 2 * pi + phase_true) + amplitude2_true * sin(modelx / period2_true * 2 * pi + phase2_true)
    x = rng.uniform(0, 360, Ndata)
    y = rng.normal(offset_true + amplitude_true * sin(x / period_true * 2 * pi + phase_true) + amplitude2_true * sin(x / period2_true * 2 * pi + phase2_true), jitter_true)
    data = dict(
        modelx = modelx,
        modely = modely,
        datax = x,
        datay = y,
        yerr = jitter_true,
    )
    
    def loglike(params):
        offset, jitter = params[:,:2].transpose()
        
        amplitudes, phases, periods = params[:,2::3].transpose(), params[:,3::3].transpose(), params[:,4::3].transpose()
        predicty = offset.reshape((-1, 1))
        for amplitude, phase, period in zip(amplitudes, phases, periods):
            predicty = predicty + amplitude.reshape((-1, 1)) * sin(x.reshape((1, -1)) / period.reshape((-1, 1)) * 2 * pi + phase.reshape((-1, 1)))
        
        jitter = jitter.reshape((-1, 1))
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y) / jitter)**2).sum(axis=1)
        assert logl.shape == (len(params),), (params.shape, offset.shape, predicty.shape, x.shape, jitter.shape)
        return logl
    
    def transform(x):
        z = x.copy()
        z[:,0] = x[:,0] * 100 - 50
        z[:,1] = 10**(x[:,1] * 1 - 1.5)
        if x.shape[1] > 2:
            z[:,2::3] = 10**(x[:,2::3] * 4 - 2)
            z[:,3::3] = 2 * pi * x[:,3::3]
            z[:,4::3] = 10**(x[:,4::3] * 4 - 1)
        return z
    
    return paramnames, loglike, transform, True, data

def get_box(ndim):
    """High box on a Gaussian."""
    
    def loglike(theta):
        delta = np.max(theta, axis=1)
        return -0.5 * ((theta / 0.1)**2).sum(axis=1) + 100 * (delta < 0.1)

    def transform(u):
        return u
    
    return autoparamnames(ndim), loglike, transform, True, None


def get_spike_and_slab(ndim, factor, offset=0, weight1=1):
    """Mixture of two gaussians.

    The narrower gaussian has standard deviation sigma2, and 
    shifted by offset in each direction.
    The wider gaussian has standard deviation sigma1=1.0.
    """
    sigma2 = 1.0 * factor**(-1. / ndim)
    # print("factor:", factor, "sigma:", sigma2)
    sigma1 = 1.0
    assert sigma2 < sigma1, (sigma2, sigma1)
    logconst1 = np.log(weight1 / (1 + weight1)) - 0.5 * np.log(2 * np.pi * sigma1**2) * ndim
    logconst2 = np.log(1 / (1 + weight1)) - 0.5 * np.log(2 * np.pi * sigma2**2) * ndim

    def loglike(theta):
        logL1 = logconst1 - 0.5 * (((theta - offset * sigma1)**2).sum() / sigma1**2)
        logL2 = logconst2 - 0.5 * (theta**2).sum() / sigma2**2
        L = np.logaddexp(logL1, logL2)
        if not L > -1e300:
            return -1e300
        return L

    def transform(u):
        return u * 20 - 10

    return autoparamnames(ndim), loglike, transform, False, None


problems = [
    ('multisine-0comp-2d', get_multisine(0)),
    ('multisine-1comp-5d', get_multisine(1)),
    ('multisine-2comp-8d', get_multisine(2)),
    ('multisine-3comp-11d', get_multisine(3)),
    ('asymgauss-4d', get_asymgauss(4)),
    ('asymgauss-16d', get_asymgauss(16)),
    ('asymgauss-100d', get_asymgauss(100)),
    ('corrfunnel-2d', get_stdfunnel(1, gamma=0.95)),
    ('corrfunnel-10d', get_stdfunnel(9, gamma=0.95)),
    ('corrfunnel-50d', get_stdfunnel(49, gamma=0.95)),
    ('corrfunnel2-2d', get_stdfunnel2(1, gamma=0.95)),
    ('corrfunnel2-10d', get_stdfunnel2(9, gamma=0.95)),
    ('corrfunnel2-50d', get_stdfunnel2(49, gamma=0.95)),
    ('corrfunnel3-2d', get_stdfunnel3(1, gamma=0.95)),
    ('corrfunnel3-10d', get_stdfunnel3(9, gamma=0.95)),
    ('corrfunnel3-50d', get_stdfunnel3(49, gamma=0.95)),
    ('corrfunnel4-2d', get_stdfunnel4(1, gamma=0.95)),
    ('corrfunnel4-10d', get_stdfunnel4(9, gamma=0.95)),
    ('corrfunnel4-50d', get_stdfunnel4(49, gamma=0.95)),
    ('rosenbrock-2d', get_rosenbrock(2)),
    ('rosenbrock-20d', get_rosenbrock(20)),
    ('rosenbrock-50d', get_rosenbrock(50)),
    #('halfrosenbrock-2d', get_halfrosenbrock(2)),
    #('halfrosenbrock-4d', get_halfrosenbrock(4)),
    #('halfrosenbrock-6d', get_halfrosenbrock(6)),
    #('halfrosenbrock-8d', get_halfrosenbrock(8)),
    #('halfrosenbrock-10d', get_halfrosenbrock(10)),
    #('halfrosenbrock-20d', get_halfrosenbrock(20)),
    #('halfrosenbrock-50d', get_halfrosenbrock(50)),
    ('eggbox-2d', get_eggbox(2)),
    ('beta-2d', get_beta(2)),
    ('beta-10d', get_beta(10)),
    ('beta-30d', get_beta(30)),
    ('loggamma-2d', get_loggamma(2)),
    ('loggamma-10d', get_loggamma(10)),
    ('loggamma-30d', get_loggamma(30)),
    ('box-5d', get_box(5)),
    ('spikeslab1-2d-4', get_spike_and_slab(2, 4)),
    ('spikeslab1-2d-40', get_spike_and_slab(2, 40)),
    ('spikeslab1-2d-400', get_spike_and_slab(2, 400)),
    ('spikeslab1-2d-4000', get_spike_and_slab(2, 4000)),
    ('spikeslab40-2d-4', get_spike_and_slab(2, 4, weight1=40)),
    ('spikeslab40-2d-40', get_spike_and_slab(2, 40, weight1=40)),
    ('spikeslab40-2d-400', get_spike_and_slab(2, 400, weight1=40)),
    ('spikeslab40-2d-4000', get_spike_and_slab(2, 4000, weight1=40)),
    ('spikeslab1000-2d-4', get_spike_and_slab(2, 4, weight1=1000)),
    ('spikeslab1000-2d-40', get_spike_and_slab(2, 40, weight1=1000)),
    ('spikeslab1000-2d-400', get_spike_and_slab(2, 400, weight1=1000)),
    ('spikeslab1000-2d-4000', get_spike_and_slab(2, 4000, weight1=1000)),
    ('spikeslab1000-2d-40-off1', get_spike_and_slab(2, 40, 1, weight1=1000)),
    ('spikeslab1000-2d-40-off2', get_spike_and_slab(2, 40, 2, weight1=1000)),
    ('spikeslab1000-2d-40-off4', get_spike_and_slab(2, 40, 4, weight1=1000)),
    ('spikeslab1000-2d-40-off10', get_spike_and_slab(2, 40, 10, weight1=1000)),
    ('spikeslab40-2d-40-off1', get_spike_and_slab(2, 40, 1, weight1=40)),
    ('spikeslab40-2d-40-off2', get_spike_and_slab(2, 40, 2, weight1=40)),
    ('spikeslab40-2d-40-off4', get_spike_and_slab(2, 40, 4, weight1=40)),
    ('spikeslab40-2d-40-off10', get_spike_and_slab(2, 40, 10, weight1=40)),
    ('spikeslab1-2d-40-off1', get_spike_and_slab(2, 40, 1)),
    ('spikeslab1-2d-40-off2', get_spike_and_slab(2, 40, 2)),
    ('spikeslab1-2d-40-off4', get_spike_and_slab(2, 40, 4)),
    ('spikeslab1-2d-40-off10', get_spike_and_slab(2, 40, 10)),
]

if __name__ == '__main__':
    import os, sys
    import logging
    from autosampler import run_sampler
    problem_name = os.environ.get('PROBLEM', '')
    if problem_name not in [n for n, _ in problems]:
        print("available toy problems:", ' '.join([n for n, _ in problems]))
        sys.exit(0)
    
    log_dir = 'systematiclogs/%s/%s/' % (problem_name, os.environ['SAMPLER'])
    os.environ['LOGDIR'] = log_dir
    os.makedirs(os.environ['LOGDIR'], exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    #handler = logging.StreamHandler(sys.stdout)
    #handler.setLevel(logging.INFO)
    #formatter = logging.Formatter('[%(name)s] %(message)s')
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)



    paramnames, loglike, transform, vectorized, _ = dict(problems)[problem_name]
    run_sampler(paramnames, loglike, transform=transform, vectorized=vectorized)
