# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 19:20:37 2021

@author: gmoha
"""

from math import sqrt, pi
import numpy as np
import scipy
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.pipeline import Pipeline
#from pisa import ureg

#Build Model
model = DistributionMaker(["settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"])
parameters = list(model.params.free.names)
numOfParams = len(parameters)
#Get Analysis B Data
data_maker = Pipeline("settings/pipeline/IceCube_3y_data.cfg")
data = data_maker.get_outputs()


def make_truncated_gaussian_transform(lo, hi, mean, std):
    rv = scipy.stats.norm(mean, std)
    xlo = rv.cdf(lo)
    xhi = rv.cdf(hi)
    def gaussian_transform(x):
        return rv.ppf(x * (xhi - xlo) + xlo)
    return gaussian_transform

def make_uniform_transform(lo, hi):
    return lambda x: lo + x * (hi - lo)

#Prepare lists to speed things up
nominal_values = []
units = []
rvs = []
for param in model.params.free:
    nominal_values.append(param.value)
    units.append(param.units)
    lo, hi = [k.magnitude for k in param.range]
    print(param, lo, hi, param.prior.kind)
    if param.prior.kind == 'uniform':
        rvs.append(make_uniform_transform(lo, hi))
    elif param.prior.kind == 'gaussian':
        mean, std = param.prior.mean.magnitude, param.prior.stddev.magnitude
        print("     %s +- %s" % (mean, std))
        rvs.append(make_truncated_gaussian_transform(lo, hi, mean, std))
    else:
        assert False, ('unknown prior kind', param.prior.kind)

#Prior (gaussian or uniform)
def prior_transform(cube):
    params = cube.copy()
    for i, rv in enumerate(rvs):
        params[i] = rv(cube[i])
    return params

get_nominal_value = np.vectorize(lambda x: x.nominal_value)
get_std_dev = np.vectorize(lambda x: x.std_dev)

data_total = data.hist['total']

#Log Likelihood
def log_likelihood(params_in):
    # create a list with units
    #Update parameters in model
    model.set_free_params([v * u for v, u in zip(params_in, units)])

    #Model predicions given params
    fit = model.get_outputs(return_sum=True)
    #Format fit to compare to data
    total = fit.hist['total']
    fit_values = get_nominal_value(total)
    # Add sqrt(N) to error (poisson)
    fit_errors = np.sqrt(get_std_dev(total)**2 + fit_values)

    #Likelihood of data points given params
    prob_data = (-(data_total - fit_values)**2)/(2*(fit_errors**2)) - np.log(sqrt(2*pi)*fit_errors)

    #Log likelihood of all data points combined
    loglike = prob_data.sum()

    return loglike


#Start sampling process

#import ultranest
#sampler = ultranest.ReactiveNestedSampler(parameters, log_likelihood, prior_transform, resume=True, log_dir='PISA-1')
#results = sampler.run()
#sampler.print_results()

#import snowline
#sampler = snowline.ReactiveImportanceSampler(parameters, log_likelihood, prior_transform)
#results = sampler.run()
#sampler.print_results()

import os
import logging
from autosampler import run_sampler
problem_name = 'icecube'
log_dir = 'systematiclogs/%s/%s/' % (problem_name, os.environ['SAMPLER'])
os.environ['LOGDIR'] = log_dir
os.makedirs(os.environ['LOGDIR'], exist_ok=True)

logging.getLogger('pisa').setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
formatter = logging.Formatter(
    '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

run_sampler(parameters, log_likelihood, transform=prior_transform)
