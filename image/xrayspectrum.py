import argparse
import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt
import scipy.stats

def main(args):
    #Ndata = args.ndata
    powerlaw_true = 2
    nh_true = 100
    fscat_true = 0.04
    amplitude_true = args.contrast
    background_true = args.background
    np.random.seed(int(args.contrast))
    paramnames = ['logamplitude', 'photonindex', 'lognh', 'logfscat', 'background']
    ndim = len(paramnames)
    
    E = np.linspace(0.5, 8, 200)
    sensitivity = exp(-np.abs((E-2)/2.))
    model = amplitude_true * E**-powerlaw_true * (fscat_true + exp(-nh_true * E**-3))
    model_convolved = model * sensitivity + background_true
    
    if True:
        plt.plot(E, model, label='intrinsic model')
        plt.plot(E, model * 0 + background_true, label='background')
        plt.plot(E, model_convolved, label='convolved model with background')
        plt.plot(E, sensitivity, color='gray', label='instrument sensitivity')
        data = np.random.poisson(model_convolved)
        plt.plot(E, data, 'x ', label='data')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='best')
        plt.savefig('xrayspectrum.pdf', bbox_inches='tight')
        plt.close()
    
    
    def loglike(params):
        logamplitude, photonindex, lognh, logfscat, background = params
        model = 10**logamplitude * E**-photonindex * (10**logfscat + exp(-10**lognh * E**-3))
        model_convolved = model * sensitivity + background_true
        logls = scipy.stats.poisson.logpmf(data, model_convolved)
        logls[~np.isfinite(logls)] = -1e300
        return logls.sum()
    
    def transform(x):
        z = x.copy()
        z[0] = x[0] * 10 - 5
        z[1] = scipy.stats.norm.ppf(x[1], loc=2.0, scale=0.2)
        z[2] = x[2] * 6 - 3
        z[3] = -1 - x[3] * 6
        z[4] = exp(scipy.stats.norm.ppf(x[4], loc=log(background_true), scale=0.1))
        return z

    loglike(transform(np.ones((ndim))*0.5))
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
        
        result = solve(LogLikelihood=loglike, Prior=transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
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
            log_dir=args.log_dir, resume='overwrite')
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir, resume='overwrite')
    
    sampler.run(frac_remain=0.5, min_num_live_points=args.num_live_points)
    
    print()
    sampler.plot()
    
    for i, p in enumerate(paramnames):
        v = sampler.results['samples'][:,i]
        print('%20s: %5.3f +- %5.3f' % (p, v.mean(), v.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal level")
    parser.add_argument('--background', type=float, default=0.2,
                        help="Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='logs/xrayspectrum')
    parser.add_argument('--reactive', action='store_true', default=False)
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)

    args = parser.parse_args()
    main(args)
