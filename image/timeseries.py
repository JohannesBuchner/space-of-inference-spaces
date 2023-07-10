import argparse
import numpy as np
from numpy import pi, sin, log
import matplotlib.pyplot as plt
import os

def main(args):
    Ndata = args.ndata
    adaptive_nsteps = args.adapt_steps
    np.random.seed(Ndata)
    jitter_true = 0.1
    phase_true = np.random.uniform(0, 2 * pi)
    period_true = 10**np.random.uniform(np.log10(1), np.log10(360*3))
    amplitude_true = args.contrast / Ndata * jitter_true
    paramnames = ['amplitude', 'jitter', 'phase', 'period']
    ndim = len(paramnames)

    log_dir = '%s-ndata%d-contrast%d' % (args.log_dir, Ndata, args.contrast)
    os.makedirs(log_dir, exist_ok=True)

    Ndaysmax = 360
    x_l = np.linspace(0, Ndaysmax, 1000)
    y_l = amplitude_true * sin(x_l / period_true * 2 * pi + phase_true)

    plt.title("amplitude: %.2f jitter: %.2f phase: %.2f period: %.2f" % (
        amplitude_true, jitter_true, phase_true, period_true))
    plt.plot(x_l, y_l)
    x = np.random.uniform(0, Ndaysmax, Ndata)
    y = np.random.normal(amplitude_true * sin(x / period_true * 2 * pi + phase_true), jitter_true)
    plt.errorbar(x, y, yerr=jitter_true, marker='x', ls=' ')
    plt.savefig(log_dir + '/input.pdf', bbox_inches='tight')
    plt.close()


    def loglike(params):
        amplitude, jitter, phase, period = params.transpose()[:4]
        predicty = amplitude * sin(x.reshape((-1,1)) / period * 2 * pi + phase)
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y.reshape((-1,1))) / jitter)**2).sum(axis=0)
        return logl

    def transform(x):
        z = np.empty((len(x), 4))
        z[:,0] = 10**(x[:,0] * 4 - 2)
        z[:,1] = 10**(x[:,1] * 1 - 1.5)
        z[:,2] = 2 * pi * x[:,2]
        z[:,3] = 10**(x[:,3] * 4 - 1)
        #z[:,4] = 2 * pi / x[:,3]
        return z

    loglike(transform(np.ones((2, ndim))*0.5))
    if args.pymultinest:
        from pymultinest.solve import solve
        global Lmax
        Lmax = -np.inf
        
        def flat_loglike(theta):
            L = loglike(theta.reshape((1, -1)))[0]
            global Lmax
            if L > Lmax:
                print("Like: %.2f" % L)
                Lmax = L
            return L
        
        def flat_transform(cube):
            return transform(cube.reshape((1, -1)))[0]
        
        result = solve(LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=log_dir,
            n_live_points=args.num_live_points,
            verbose=True, resume=False, importance_nested_sampling=False)
        
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
        return
    
    elif args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=log_dir, vectorized=True, resume=True)
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps)
        sampler.run(min_num_live_points=args.num_live_points)
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            log_dir=log_dir, vectorized=True, resume='overwrite')
    
        sampler.run(num_live_points=args.num_live_points)

    print()
    sampler.plot()

    for i, p in enumerate(paramnames):
        v = sampler.results['samples'][:,i]
        print('%20s: %5.3f +- %5.3f' % (p, v.mean(), v.std()))

    amplitude, jitter, phase, period = sampler.results['samples'].transpose()
    plt.title("amplitude: %.2f jitter: %.2f phase: %.2f period: %.2f" % (
        amplitude.mean(), jitter.mean(), phase.mean(), period.mean()))

    from ultranest.plot import PredictionBand
    plt.errorbar(x, y, yerr=jitter_true, marker='x', ls=' ')
    band = PredictionBand(x_l)
    for amplitude, jitter, phase, period in sampler.results['samples']:
        y_pred = amplitude * sin(x_l / period * 2 * pi + phase)
        band.add(y_pred)
    band.line(color='gray')
    band.shade(color='gray', alpha=0.2)

    plt.savefig(log_dir + '/output.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal-to-Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs/sine')
    parser.add_argument('--reactive', action='store_true', default=False)
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)

    args = parser.parse_args()
    main(args)
