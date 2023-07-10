import os
import logging
import argparse
import numpy as np
from numpy import exp, log, log10
import matplotlib.pyplot as plt
import scipy.stats
from autosampler import run_sampler

def smoothlybrokenpowerlaw(x, K, pivot, break_energy, break_scale, low_index, high_index):

    B = (low_index + high_index) / 2.0
    M = (high_index - low_index) / 2.0

    arg_piv = np.log10(pivot / break_energy) / break_scale

    if arg_piv < -6.0:
        pcosh_piv = M * break_scale * (-arg_piv - np.log(2.0))
    elif arg_piv > 4.0:

        pcosh_piv = M * break_scale * (arg_piv - np.log(2.0))
    else:
        pcosh_piv = M * break_scale * (np.log((np.exp(arg_piv) + np.exp(-arg_piv)) / 2.0))

    arg = np.log10(x / break_energy) / break_scale
    idx1 = arg < -6.0
    idx2 = arg > 4.0
    idx3 = ~np.logical_or(idx1, idx2)

    # The K * 0 part is a trick so that out will have the right units (if the input
    # has units)

    pcosh = np.zeros(x.shape)

    pcosh[idx1] = M * break_scale * (-arg[idx1] - np.log(2.0))
    pcosh[idx2] = M * break_scale * (arg[idx2] - np.log(2.0))
    pcosh[idx3] = M * break_scale * (np.log((np.exp(arg[idx3]) + np.exp(-arg[idx3])) / 2.0))

    return K * (x / pivot) ** B * 10. ** (pcosh - pcosh_piv)



def main(args):
    #Ndata = args.ndata
    powerlaw_true = 2
    nh_true = 100
    fscat_true = 0.04
    amplitude_true = args.contrast
    background_true = args.background
    np.random.seed(int(args.contrast))
    paramnames = ['logamplitude', 'photonindex', 'lognh', 'logfscat', 'background',
        'amplitude', 'break_energy', 'break_scale', 'high_index']

    E = np.linspace(0.5, 8, 200)
    sensitivity = exp(-0.5 * (log10(E / 1.)/0.25)**2)
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
        plt.xlabel('Energy [keV]')
        plt.ylabel('Count rate [Counts / s / keV]')
        plt.legend(loc='best')
        plt.savefig('bixrayspectrum-%d.pdf' % args.contrast, bbox_inches='tight')
        plt.close()

    E2 = np.logspace(2, 3, 20)
    model2 = smoothlybrokenpowerlaw(E2, amplitude_true, 1.0, 400.0, 0.25, -(powerlaw_true - 2.), -2.)
    sensitivity2 = exp(-0.5*((log10(E2/500.))/0.2)**2) * 3
    model2_convolved = model2 * sensitivity2

    plt.plot(E2, model2, label='intrinsic model')
    plt.plot(E2, sensitivity2, color='gray', label='instrument sensitivity')
    plt.plot(E2, model2_convolved, label='convolved model')
    #plt.plot(E2, model3_convolved, label='convolved model with background, no cutoff')
    data2 = np.random.poisson(model2_convolved)
    plt.plot(E2, data2, 'x ', label='data')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Count rate $\times$ E$^2$ [keV$^2$ $\times$ Counts / s / keV]')
    plt.legend(loc='best')
    plt.savefig('bixrayspectrum-%d_bending.pdf' % args.contrast, bbox_inches='tight')
    plt.close()

    def loglike(params):
        logamplitude, photonindex, lognh, logfscat, background, logK, break_energy, break_scale, high_index = params
        model = 10**logamplitude * E**-photonindex * (10**logfscat + exp(-10**lognh * E**-3))
        model_convolved = model * sensitivity + background_true
        logls = scipy.stats.poisson.logpmf(data, model_convolved)
        logls[~np.isfinite(logls)] = -1e300

        low_index = -(photonindex - 2)
        model2 = smoothlybrokenpowerlaw(E2, 10**logK, 1.0, break_energy, break_scale, low_index, high_index)
        model2_convolved = model2 * sensitivity2
        logls2 = scipy.stats.poisson.logpmf(data2, model2_convolved)
        logls2[~np.isfinite(logls2)] = -1e300
        return logls.sum() + logls2.sum()

    def transform(x):
        z = x.copy()
        z[0] = x[0] * 10 - 5
        z[1] = scipy.stats.norm.ppf(x[1], loc=2.0, scale=0.2)
        z[2] = x[2] * 6 - 3
        z[3] = -1 - x[3] * 6
        z[4] = exp(scipy.stats.norm.ppf(x[4], loc=log(background_true), scale=0.1))
        z[5] = scipy.stats.norm.ppf(x[5], loc=z[0], scale=0.5)
        #z[5] = x[5] * 4 - 2
        z[6] = 10**(x[6] * 2 + 1) # 10 ... 1000
        z[7] = 10**(x[7] * 2 - 2) # 0.01 .. 1
        z[8] = x[8] * 5 - 5
        #z[9] = x[9] * 8 - 4
        return z
    
    problem_name = 'bixrayspectrum-%d' % args.contrast
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

    print("running sampler ...")
    run_sampler(paramnames, loglike, transform=transform)
    print("running sampler done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal level")
    parser.add_argument('--background', type=float, default=0.2,
                        help="Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")

    args = parser.parse_args()
    main(args)
