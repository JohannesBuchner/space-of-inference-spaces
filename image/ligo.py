import os
import logging

def main():
	# based on several tutorials, but primarily
	# https://nbviewer.org/github/gwastro/pycbc-tutorials/blob/master/tutorial/inference_1_ModelsAndPEByHand.ipynb
	
	from pycbc.psd import interpolate, inverse_spectrum_truncation
	from pycbc.frame import read_frame
	from pycbc.filter import highpass, resample_to_delta_t
	from pycbc.catalog import Merger
	from astropy.utils.data import download_file

	m = Merger("GW170817")

	# List of observatories we'll analyze
	ifos = ['H1', 
			'V1', 
			'L1',
		   ]

	# The single template waveform model needs these data products
	psds = {}
	data = {}

	for ifo in ifos:
		print("Processing {} data".format(ifo))
		
		# Download the gravitational wave data for GW170817
		url = "https://dcc.ligo.org/public/0146/P1700349/001/{}-{}1_LOSC_CLN_4_V1-1187007040-2048.gwf"
		fname = download_file(url.format(ifo[0], ifo[0]), cache=True) 

		# Read the gravitational wave data and do some minimal
		# conditioning of the data.
		ts = read_frame(fname, "{}:LOSC-STRAIN".format(ifo),
						start_time=int(m.time - 260),
						end_time=int(m.time + 40))
		ts = highpass(ts, 15.0)                     # Remove low frequency content
		ts = resample_to_delta_t(ts, 1.0/2048)      # Resample data to 2048 Hz
		ts = ts.time_slice(m.time-112, m.time + 16) # Limit to times around the signal
		data[ifo] = ts.to_frequencyseries()         # Convert to a frequency series by taking the data's FFT

		# Estimate the power spectral density of the data
		psd = interpolate(ts.psd(4), ts.delta_f)
		psd = inverse_spectrum_truncation(psd, int(4 * psd.sample_rate), 
										  trunc_method='hann',
										  low_frequency_cutoff=20.0)
		psds[ifo] = psd



	from pycbc.inference import models
	from pycbc.distributions import Uniform, JointDistribution, UniformAngle, SinAngle, CosAngle

	static = {'f_lower':25.0,
			  'approximant':"TaylorF2",
			  'polarization':0,
			  'f_final':500,
			 }

	variable = ('mchirp',
				'q',
				'tc',
				'distance',
				'inclination',
				'ra',
				'dec',
				)
	prior = JointDistribution(variable, 
					SinAngle(inclination=None),
					UniformAngle(ra=None),
					CosAngle(dec=None),
					Uniform(
						distance=(10, 100),
						mchirp=(1.0, 2.0),
						q=(1, 2.),
						tc=(m.time+.02, m.time+0.05),
					),
			)

	from pycbc.transforms import MchirpQToMass1Mass2

	model = models.MarginalizedPhaseGaussianNoise(variable, data,
												  low_frequency_cutoff = {'H1':25, 'L1':25, 'V1':25},
												  high_frequency_cutoff = {'H1':500, 'L1':500, 'V1':500},
												  psds = psds,
												  static_params = static,
												  prior = prior,
												  waveform_transforms = [MchirpQToMass1Mass2()]
												 )

	print("setting up sampler...")
	from pycbc.inference.sampler.multinest import MultinestSampler
	smpl = MultinestSampler(model, nlivepoints=400)
	log_likelihood_call = smpl.loglikelihood
	def prior_call(u):
		x = u.copy()
		smpl.transform_prior(x)
		return x

	from autosampler import run_sampler

	problem_name = 'ligo'
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
	run_sampler(list(model.variable_params), log_likelihood_call, transform=prior_call)
	print("running sampler done")

if __name__ == '__main__':
	main()
