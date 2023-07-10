================================
Problem set for Nested Sampling
================================

A set of test problems for nested sampling which test:

* low & high dimensionality (2-100 parameters)
* multi-modality (1-10 peaks)
* asymmetry (one parameter may be 10,000 times better constrained than another)
* non-gaussianity (degeneracies can look like bananas or boomerangs)
* heavy tails (not everything declines outwards like a gaussian)
* depth (how small is the posterior compared to the prior tells how much information is in the data)


A full list of problems is given in problems.txt, but they include:

Physics
-------

 * astroparticle pisa-icecube-neutrino
 * cosmology CMB-planck
 * gravitational waves	gravwave-ligo
 * supernova remnants	crab-gamma
 * exoplanets juliet-exo-transient systematiclogs/transit/ultranest-safe/
 * exoplanets juliet-exo-rv 0 planets systematiclogs/exoplanet-rvs_0005.txt-0/ultranest-safe
 * exoplanets juliet-exo-rv 1 planets systematiclogs/exoplanet-rvs_0005.txt-1/ultranest-safe
 * exoplanets juliet-exo-rv 2 planets systematiclogs/exoplanet-rvs_0005.txt-2/ultranest-fast-fixed4d
 * exoplanets juliet-exo-rv 3 planets systematiclogs/exoplanet-rvs_0005.txt-3/ultranest-fast-fixed4d
 * transients mosfit-SLSN-LSQ12dlf systematiclogs/mosfit-LSQ12dlf/ultranest-safe/
 * transients mosfit-magnetar-LSQ12dlf systematiclogs/mosfit-LSQ12dlf-magnetar/ultranest-safe/
 * supernova-remnants 3ML-crab systematiclogs/crab/ultranest/
 * extragalactic BXA-xray-agn-spec-fit-CDFS-179 systematiclogs/xrayagnspec/ultranest-safe/
 * materials Lennard-Jones potential with 6 particles systematiclogs/LJ6/
 * gravitational-waves ligo-GW: systematiclogs/ligo/ultranest-fast-fixed4d/
 * extragalactic line: systematiclogs/line/ultranest-safe/ 
 * extragalactic gaussdist: systematiclogs/posteriorstacker/posteriorstacker-flex11/ 
 * extragalactic histdist: systematiclogs/posteriorstacker/posteriorstacker-gauss/ 


Mock data
---------

 * Compton-thick-AGN systematiclogs/xrayspectrum40-0.01/ 
 * multisine-{0,1,2,3}comp
	systematiclogs/multisine-0comp-2d/ultranest-safe/
	systematiclogs/multisine-1comp-5d/ultranest-safe/
	systematiclogs/multisine-2comp-8d/ultranest-fast-fixed4d/
	systematiclogs/multisine-3comp-11d/ultranest-fast-fixed4d/

Toy 
---

 * asymgauss-4,16,100d
	systematiclogs/asymgauss-4d/ultranest-safe/
	systematiclogs/asymgauss-100d/ultranest-fast-fixed4d/
	systematiclogs/asymgauss-16d/ultranest-fast-fixed4d/
 * beta-2,10,30d
	systematiclogs/beta-10d/ultranest-safe/
	systematiclogs/beta-2d/ultranest-safe/
	systematiclogs/beta-30d/ultranest-fast-fixed4d/
 * box-5d
	systematiclogs/box-5d/ultranest-safe/
 * eggbox-2d
	systematiclogs/eggbox-2d/ultranest-safe/
 * loggamma-2,10,30d
	systematiclogs/loggamma-10d/ultranest-safe/
	systematiclogs/loggamma-2d/ultranest-safe/
	systematiclogs/loggamma-30d/ultranest-fast-fixed4d/
 * rosenbrock-2,20,50d
	systematiclogs/rosenbrock-2d/ultranest-safe/
	systematiclogs/rosenbrock-20d/ultranest-safe/
	systematiclogs/rosenbrock-50d/ultranest-fast-fixed4d/
 * corrfunnel4-2,10,50d
	systematiclogs/corrfunnel4-10d/ultranest-safe/
	systematiclogs/corrfunnel4-2d/ultranest-safe/
	systematiclogs/corrfunnel4-50d/ultranest-fast-fixed4d/


Build
------

The code pipelines that support the various inference problems are extensive and
complex.
To allow a reproducible build that runs anywhere, a combination of conda and docker is used.
The image/Dockerfile tells docker how to build an image (similar to a virtual machine) 
that can be entered.
To build the docker image, and install all the software in it.

In the image/ folder, run::

	$ docker build .

This should result in a hash (such as 93a2fd7459c8). You need it for the next step.

make a folder systematiclogs/ where the results will be stored::

	mkdir systematiclogs

To enter the compute image with all likelihoods, and link the systematiclogs folder to your computer, run::

	$ docker run -it -v $PWD/systematiclogs:/root/systematiclogs <HASH> bash

where <HASH> is the hash of the build step.

Test
------

To test, you can run the test sampler::

	bash runtoy.sh testsampler
	bash runreal.sh testsampler

This is also run by default, and can be done in isolation:

	docker run -it <HASH>

This takes a while...

Run
------

The autosampler.py allows swapping out various samplers.
The desired sampler is chosen by the environment SAMPLER.
You may want to add your sampler to autosampler.py.

Toy problems, some with analytic solutions, are implemented in pure python
in problems.py.
Inside the docker image, to run one toy problem::

	SAMPLER=ultranest-safe PROBLEM=beta-2d python3 problems.py 

The runtoy.sh script automates this.

To run real problems, see runreal.sh

Some problems need to be run through their own interfaces,
and modifications to those code bases were necessary to hook in the UltraNest sampler.
The commands are documented in runreal.sh.
These are currently: cosmology-CMB (montepython), mosfit, posteriorstacker, 3ML (crab.py, grb.py)

TODO
----

Help is welcome! We need you!

* If you run into any issues, please open a github issue!
* Can you provide ground truth results from fine integrations?
* Maybe add your own sampler? Which problems can it run?
* Can you add a deep learning emulator to accelerate the very slow likelihood of icecube?
* Can we approximate nested sampling run outputs with a fast, analytic model, so that we have a very similar inference problem but know the truth? For example, approximate with a gaussian mixture model or a deep neural network that predicts the log-likelihood?
* Can you help bring autosampler into montepython (or cobaya), 3ML, posteriorstacker so we have a uniform interface there as well?
* Do you have additional toy problems that cover a new parameter space that should be added? 
  Ideally it should be motivated by some physics problem.
* If you would like to help, but are unsure how, send me an email or open a github issue.

More information
----------------


A draft paper is in pres/
