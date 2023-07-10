import os
import numpy as np
import juliet
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--rvfile', type=str, 
	default='exoplanet/rvs_0005.txt',
	help="Radial velocity data file")
parser.add_argument('--planets', type=int, default=0,
					help="Number of planets")
parser.add_argument("--num_live_points", type=int, default=400)
#parser.add_argument('--reactive', action='store_true', default=False)
#parser.add_argument('--pymultinest', action='store_true')
#parser.add_argument('--slice_steps', type=int, default=100)
#parser.add_argument('--adapt_steps', type=str)

args = parser.parse_args()

datafile = args.rvfile
nplanets = args.planets

problem_name = 'exoplanet-%s-%d' % (os.path.basename(datafile), nplanets)
log_dir = 'systematiclogs/%s/%s/' % (problem_name, os.environ['SAMPLER'])
os.environ['LOGDIR'] = log_dir
os.makedirs(log_dir, exist_ok=True)

t = np.array([float(line.split()[0]) for line in open(datafile).readlines()])

dataset = juliet.load(
	priors = 'exoplanet/parameters%d.txt' % nplanets, 
	rvfilename=datafile, 
	out_folder = log_dir,
	GP_regressors_rv=dict(sim=t),
)

results = dataset.fit(n_live_points=args.num_live_points, 
	require_planet_periods_ordered=1.2, use_ultranest=True)
