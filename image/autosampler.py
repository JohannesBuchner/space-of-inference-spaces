"""Calculates the Bayesian evidence and posterior samples of arbitrary monomodal models."""

from __future__ import print_function
from __future__ import division

import time
import os
import numpy as np
import corner
import matplotlib.pyplot as plt
import json


class RegionLogger():
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        filename = self.prefix + 'region_%d.txt.gz' % info['it']
        np.savetxt(filename, region.u)


def ultranest_run(sampler, log_dir, max_ncalls, **kwargs):
    offset = 0
    # first quick run-through:
    for i, results_ in enumerate(sampler.run_iter(
        frac_remain=0.5, max_num_improvement_loops=0, 
        max_ncalls=max_ncalls, viz_callback=RegionLogger(log_dir), **kwargs)
    ):
        results = dict(results_)
        samples = results.pop('samples')
        del results['weighted_samples']
        param_names = results['paramnames']
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))
        offset = i + 1

    for i, results_ in enumerate(sampler.run_iter(max_ncalls=max_ncalls, viz_callback=RegionLogger(log_dir), **kwargs), start=offset):
        # complete run-through, with potential reactive iterations
        results = dict(results_)
        samples = results.pop('samples')
        del results['weighted_samples']
        param_names = results['paramnames']
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

    with open(log_dir + '/results.json', 'w') as fout:
        json.dump(results, fout, indent=4)
    np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))

    return results_


def run_sampler(
    param_names,
    loglike,
    transform=None,
    vectorized=False,
    max_ncalls=400000000
):
    """Initialise sampler.

    Parameters
    -----------
    param_names: list of str, names of the parameters.
        Length gives dimensionality of the sampling problem.

    loglike: function
        log-likelihood function.
        Receives multiple parameter vectors, returns vector of likelihood.
    transform: function
        parameter transform from unit cube to physical parameters.
        Receives multiple cube vectors, returns multiple parameter vectors.
    vectorized: bool
        if true, likelihood and transform receive arrays of points, and return arrays
    max_ncalls: int
        maximum number of likelihood evaluations
    """
    samplername = os.environ['SAMPLER']
    log_dir = os.environ['LOGDIR']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    ndim = len(param_names)
    
    if vectorized:
        def flat_loglike(theta):
            L = loglike(theta.reshape((1, -1)))[0]
            assert np.isfinite(L), L
            return L
        
        def flat_transform(cube):
            return transform(cube.reshape((1, -1)))[0]
    else:
        flat_transform = transform
        flat_loglike = loglike

    if samplername == 'testsampler':
        for i in range(2):
            print("test sampler: drawing one prior sample")
            tstart = time.time()
            u = np.random.uniform(size=ndim)
            print("  transforming unit cube sample:", u)
            p = flat_transform(u)
            print("  likelihood evaluation of parameters:", p)
            L = flat_loglike(p)
            print("  log-likelihood is:", L)
            print("  --> computational cost: %.5f seconds" % (time.time() - tstart))
    elif samplername == 'multinest':
        from pymultinest.solve import solve
        import pymultinest
        results = solve(
            LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=log_dir,
            n_live_points=400,
            verbose=True, resume=True, importance_nested_sampling=False,
            sampling_efficiency=0.3)
        analyser = pymultinest.Analyzer(ndim, log_dir, verbose=False)
        samples = analyser.get_equal_weighted_posterior()[:,:-1]
        info = analyser.get_stats()
        info['ncall'] = int(open(log_dir + '/resume.dat').readlines()[1].split()[1])
        info['maximum_likelihood'] = analyser.get_best_fit()
        info['logz'] = info['nested sampling global log-evidence']
        info['logzerr'] = info['nested sampling global log-evidence error']
        json.dump(info, open(log_dir + '/results.json', 'w'), indent=4)
        if ndim <= 30:
            corner.corner(
                results['samples'],
                labels=param_names,
                show_titles=True)
            plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
            plt.close()
    elif samplername == 'nestle':
        import nestle
        res = nestle.sample(
            flat_loglike, flat_transform, ndim,
            npoints=400, method='multi',
            callback=nestle.print_progress, maxcall=max_ncalls)
        results = dict(
            ncall=int(res.ncall),
            logz=float(res.logz),
            logzerr=float(res.logzerr),
        )
        weighted_samples, weights = res.samples, res.weights
        assert np.isclose(weights.sum(), 1), (weights.max(), weights.sum(), weights)
        samples = nestle.resample_equal(weighted_samples, weights)
        with open(log_dir + '/results.txt', 'w') as f:
            f.write(str(results))
        try:
            with open(log_dir + '/results.json', 'w') as f:
                json.dump(results, f, indent=4)
        except Exception:
            pass
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        corner.corner(
            samples,
            labels=param_names,
            show_titles=True)
        plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
        plt.close()
    elif samplername == 'dynesty' or samplername == 'dynesty-multiell':
        from dynesty import DynamicNestedSampler
        from dynesty import utils as dyfunc
        if samplername == 'dynesty-multiell':
            sampler = DynamicNestedSampler(flat_loglike, flat_transform, ndim,
                bound='multi', sample='unif')
        else:
            sampler = DynamicNestedSampler(flat_loglike, flat_transform, ndim)
        sampler.run_nested(maxcall=max_ncalls, nlive_init=100, maxiter_init=10, nlive_batch=100)
        res = sampler.results
        try:
            print("sampler summary:")
            sampler.summary()
        except Exception:
            pass
        try:
            print("results summary:")
            res.summary()
        except Exception:
            pass
        results = dict(
            ncall=int(sum(res.ncall)),
            logz=float(res.logz[-1]),
            logzerr=float(res.logzerr[-1]),
        )
        weighted_samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
        samples = dyfunc.resample_equal(weighted_samples, weights)
        with open(log_dir + '/results.txt', 'w') as f:
            f.write(str(results))
        try:
            with open(log_dir + '/results.json', 'w') as f:
                json.dump(results, f, indent=4)
        except Exception:
            pass
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        corner.corner(
            samples,
            labels=param_names,
            show_titles=True)
        plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
        plt.close()
    elif samplername == 'ultranest-safe':
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=vectorized, ndraw_max=10000000)
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-fast-fixed100':
        from ultranest import ReactiveNestedSampler
        slice_steps = 100
        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        import ultranest.stepsampler
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=slice_steps, region_filter=False,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            log=open(log_dir + '/stepsampler.log', 'w')
        )
        
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-fast-fixed4d':
        from ultranest import ReactiveNestedSampler
        slice_steps = 4 * len(param_names)
        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        import ultranest.stepsampler
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=slice_steps, region_filter=False,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            log=open(log_dir + '/stepsampler.log', 'w')
        )
        
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-fast':
        from ultranest import ReactiveNestedSampler
        slice_steps = 100
        adaptive_nsteps = 'move-distance'
        
        # log_dir = log_dir + 'RNS-%dd-harm%d' % (ndim, slice_steps)
        if adaptive_nsteps:
            log_dir = log_dir + '-adapt%s' % (adaptive_nsteps)

        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        import ultranest.stepsampler
        sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(
            nsteps=slice_steps, adaptive_nsteps=adaptive_nsteps, region_filter=True)
        
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-faster':
        from ultranest import ReactiveNestedSampler
        from ultranest.mlfriends import RobustEllipsoidRegion
        from ultranest.popstepsampler import PopulationSliceSampler, generate_cube_oriented_direction
        nsteps = 100
        popsize = 10 if vectorized else 1
        
        # log_dir = log_dir + 'RNS-%dd-unit%d' % (ndim, nsteps)

        sampler = ReactiveNestedSampler(
            param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        
        sampler.stepsampler = PopulationSliceSampler(
            popsize=popsize, nsteps=nsteps,
            generate_direction=generate_cube_oriented_direction, log=False
        )
        
        ultranest_run(sampler, log_dir, max_ncalls, region_class=RobustEllipsoidRegion)
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-HD':
        from ultranest import ReactiveNestedSampler
        slice_steps = 100
        adaptive_nsteps = 'move-distance'
        
        log_dir = log_dir + 'RNS-%dd-aharm%d' % (ndim, slice_steps)
        if adaptive_nsteps:
            log_dir = log_dir + '-adapt%s' % (adaptive_nsteps)

        sampler = ReactiveNestedSampler(
            param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        import ultranest.stepsampler
        sampler.stepsampler = ultranest.stepsampler.AHARMSampler(
            nsteps=slice_steps, adaptive_nsteps=adaptive_nsteps, region_filter=True)
        
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    elif samplername in ('goodman-weare', 'slice'):
        from autoemcee import ReactiveAffineInvariantSampler
        sampler = ReactiveAffineInvariantSampler(
            param_names, loglike, transform=transform, 
            sampler=samplername, vectorized=vectorized)
        results = dict(sampler.run(max_ncalls=max_ncalls))
        sampler.print_results()
        samples = results.pop('samples')
        json.dump(results, open(log_dir + '/results.json', 'w'), indent=4)
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        if ndim <= 30:
            sampler.plot()
            plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
            plt.close()
    elif samplername == 'vbis' or samplername == 'vbis-wide':
        from snowline import ReactiveImportanceSampler
        sampler = ReactiveImportanceSampler(param_names, flat_loglike, transform=flat_transform)
        i = 0
        results = dict(sampler.laplace_approximate(
            num_global_samples=ndim * 1000 * (10 if samplername == 'vbis-wide' else 1)))
        samples = results.pop('samples')
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

        for i, results_ in enumerate(sampler.run_iter(
            num_gauss_samples=ndim * 200,
            max_ncalls=max_ncalls,
            min_ess=400,
            max_improvement_loops=40,
            heavytail_laplaceapprox=(samplername == 'vbis-wide'),
        ), start=1):
            results = dict(results_)
            samples = results.pop('samples')
            with open(log_dir + '/results-%d.json' % i, 'w') as fout:
                json.dump(results, fout, indent=4)
            np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

        sampler.print_results()
        json.dump(results, open(log_dir + '/results.json', 'w'), indent=4)
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        if ndim <= 30:
            corner.corner(
                samples,
                labels=param_names,
                show_titles=True)
            plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
            plt.close()
    else:
        assert False, ("unknown sampler:", samplername)

        
if __name__ == '__main__':
    
    param_names = ['Hinz', 'Kunz']

    def loglike(z):
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum() + -0.5 * ((z[0] - z[1])/0.01)**2
        return a

    def transform(x):
        return 10. * x - 5.
    
    run_sampler(param_names, loglike, transform)
