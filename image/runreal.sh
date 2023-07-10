#export PYTHONPATH=../autoemcee/:../mininest/:../snowline/:../dynesty/ 

function runreal() {
	# arg 1: sampler
	s=$1
	# arg 2: problem name
	p=$2
	# arg 3: command to run
	cmd=$3
	
	echo $s $p "[$cmd]"
	# skip if already there
	[ -e systematiclogs/$p/$s ] && return

	echo
	echo
	echo "===== PROBLEM:$p === SAMPLER:$s ===== "
	echo
	echo

	mkdir -p systematiclogs/$p/$s
	SAMPLER=$s python3 -u -O $cmd 2>&1 | tee systematiclogs/$p/$s/log.txt

}

# iterate over the command line arguments (samplers):
for s in $*
do
	runreal $s bixrayspectrum-30 bixrayspectrum.py --contrast=30  # via autosampler
	runreal $s crab crab.py   # 3ML support a few samplers
	runreal $s transit exoplanet-transit.py
	runreal $s exoplanet-rvs_0005.txt-0 "exoplanet.py --planets=0 --rvfile=exoplanet/rvs_0005.txt"
	runreal $s exoplanet-rvs_0005.txt-1 "exoplanet.py --planets=1 --rvfile=exoplanet/rvs_0005.txt"
	runreal $s exoplanet-rvs_0005.txt-2 "exoplanet.py --planets=2 --rvfile=exoplanet/rvs_0005.txt"
	runreal $s ligo ligo.py   # PyCBC supports a few samplers
	runreal $s icecube icecube.py  # via autosampler
	
	# this is broken: https://github.com/threeML/threeML/issues/622
	# runreal $s grb grb.py     # 3ML support a few samplers
done

# for cosmology example:
# python montepython/MontePython.py run -m UN -o systematiclogs/cmb-cosmology/ultranest/ -p input/example_ns.param

# for slsn and magnetar analyses of LSQ12dlf:
# mosfit -m slsn -e LSQ12dlf --no-copy-at-launch --method=ultranest -o systematiclogs/mosfit-LSQ12dlf/ultranest/
# mosfit -m magnetar -e LSQ12dlf --no-copy-at-launch --method=ultranest -o systematiclogs/mosfit-LSQ12dlf-magnetar/ultranest/

# for posteriorstacker-gauss and posteriorstacker-flex11
# cd PosteriorStacker; python PosteriorStacker/run.py
