import sys
import numpy as np
import ultranest
import ultranest.stepsampler
import ultranest.mlfriends
#import matplotlib.pyplot as plt

num_particles = int(sys.argv[1])

paramnames = ["z2"]
for i in range(3,num_particles+1):
    if i == 3:
        paramnames += ["y%d" % i, "z%d" % i]
    else:
        paramnames += ["x%d" % i, "y%d" % i, "z%d" % i]

sigma = 1e-3
# translation symmetry (3 parameters fewer)
# translate to center on first particle
pos0 = np.zeros(3) # pin x1=y1=z1=0
# rotation symmetry (3 parameters fewer)
# rotate to have second particle on z axis
pos1 = np.zeros(2) # pin x2=y2=0
# rotate to have third particle on x axis
pos2 = np.zeros(1) # pin x3=0

def loglikelihood(param):
    logL = 0
    if len(param) > 1:
        coordinates = np.hstack((pos0, pos1, param[:1], pos2, param[1:])).reshape((-1, 3))
    else:
        coordinates = np.hstack((pos0, pos1, param[:1])).reshape((-1, 3))

    # require increasing z indices
    #coordinates[:,3]np.cumsum(coordinates[:,3])
    unordered = np.diff(np.abs(coordinates[:,2])) < 0
    if np.any(unordered):
        #print('z coordinates:', coordinates[:,2], "rejected", np.diff(np.abs(coordinates[:,2])))
        return -1e200 * np.max(-np.diff(np.abs(coordinates[:,2])))

    for i, a in enumerate(coordinates):
        b = coordinates[i+1:,:]
        r = ((a.reshape((1, 3)) - b)**2).sum(axis=1)**0.5
        assert len(r) == len(b), (r.shape, b.shape)
        r[r < sigma] = sigma
        sigma_r6 = (sigma / r)**6
        logL += np.log(sigma_r6 - sigma_r6**2 + 1e-100).sum()
    
    return logL

def prior(cube):
    param = 2 * cube - 1
    # reflection symmetry:
    # x, y, z axes can be flipped
    # and therefore are assumed positive here
    param[:2] = cube[:2]
    if len(cube) > 2:
        param[3] = cube[3]
    
    return param


sampler = ultranest.ReactiveNestedSampler(paramnames, loglikelihood, transform=prior,
	log_dir='LJ%d' % num_particles, resume=True, region_class = ultranest.mlfriends.RobustEllipsoidRegion)
#results = sampler.run(max_ncalls=400000)
sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    40, generate_direction=ultranest.stepsampler.generate_mixture_random_direction
)
sampler.run()
sampler.print_results()
sampler.plot()
