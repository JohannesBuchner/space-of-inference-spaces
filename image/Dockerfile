# Prediction of host masses from GRB X-ray afterglows

# you need to pass the environment variable $i = Swift Trigger ID
# and mount /opt/output/ to a directory

FROM continuumio/miniconda3:latest

MAINTAINER johannesbuchner

LABEL description="integration-problem-set"

RUN apt-get update -y
RUN apt-get install -y git build-essential gfortran libblas-dev liblapack-dev

WORKDIR /root

#RUN conda update -n base -c defaults conda && conda clean --all -y
# select python 3.9 because 3.10 is not supported by fermitools yet
RUN conda install -y -c conda-forge "python=3.9.*=*_cpython" mamba && mamba clean --all -y
#RUN mamba install -y -c conda-forge ncurses && mamba clean --all -y
# ultranest and other useful libraries
RUN mamba install -y -c conda-forge numpy==1.23.5 ultranest pymultinest corner matplotlib astropy h5py getdist joblib mosfit && mamba clean --all -y

# 3ML install
RUN mamba install -y -c conda-forge -c fermi "fermitools>=2.2.0" healpy gammapy fermipy gsl && mamba clean --all -y
RUN mamba install -y -c conda-forge -c threeml astromodels threeml && mamba clean --all -y
ENV CALDB=/opt/conda/share/fermitools/data/caldb/
#RUN find /opt/conda -name alias_config.fits
ENV IRF=P8R3_SOURCE_V2
RUN ls ${CALDB}/*/
ENV FERMITOOLS=/opt/conda/share/fermitools
RUN mkdir -p /refdata/fermi; cd /refdata/fermi; ln -s ${CALDB}
RUN python -c 'from threeML import FermiLATSourceCatalog'
RUN python -c 'import ultranest'
RUN python -c 'import pymultinest'
# download cosmology requirements
RUN wget https://lesgourg.github.io/class_public/class_public-3.2.0.tar.gz  -O - | tar -xzvf -
RUN wget "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.01.tar.gz" -O - | tar -xzvf -
RUN wget "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz" -O - | tar -xzvf -

# pisa requirements
RUN mamba install -y -c conda-forge decorator "fast-histogram>=0.10" nlopt h5py iminuit numba "pint<=0.19" pandas simplejson pytables tabulate uncertainties llvmlite py-cpuinfo sympy cython && mamba clean --all
#RUN mamba install -y -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge ciao sherpa bxa && mamba clean --all

RUN make -C class_public-3.2.0 
RUN cd class_public-3.2.0/python/ && python setup.py build && python setup.py install && cd -

# that ftp server does not operate anymore, replace with https:
RUN sed --in-place 's,ftp://heasarc.gsfc.nasa.gov/,https://heasarc.gsfc.nasa.gov/FTP/,g' code/plc*/plc*/waf_tools/cfitsio.py
RUN cd code/plc*/plc* && python3 waf configure --install_all_deps && python3 waf install && cd -
# RUN wget -O - "https://github.com/brinckmann/montepython_public/archive/refs/tags/v3.5.0.tar.gz" | tar -xzvf -
RUN git clone "https://github.com/JohannesBuchner/montepython_public/"
RUN cd montepython_public/ && python montepython/MontePython.py --help && cd -

RUN pip install git+https://github.com/icecube/pisa pycbc snowline cobaya juliet 

COPY exoplanet exoplanet
COPY juliet juliet
COPY montepython_public montepython_public
#COPY cobaya cobaya
RUN pwd
RUN echo ${HOME}
RUN ls /home

RUN cd montepython_public/ && python montepython/MontePython.py --help && cd -
RUN cd /refdata/fermi; ln -s /refdata/
#RUN cd cobaya/cobaya; ls; pip install .
RUN pip uninstall -y fermipy; pip install "fermipy>=1.2"
RUN git clone https://github.com/JohannesBuchner/PosteriorStacker 
RUN cd PosteriorStacker; python PosteriorStacker/posteriorstacker.py  --help

COPY ultranest_sampler.py /opt/conda/lib/python3.9/site-packages/threeML/bayesian/ultranest_sampler.py
COPY *.py ./
COPY *.sh ./

RUN python importtester.py
RUN mkdir -p /root/modules/observables/filters/
# kilonova.space is broken
RUN sed --in-place s/kilonova\.space/tde.space/g  /opt/conda/lib/python3.9/site-packages/mosfit/fetcher.py

# default command:
# to run one toy problem:
# CMD SAMPLER=ultranest-fast-fixed4d PROBLEM=beta-2d python3 problems.py 
# or to run all toy problems:
# CMD bash runtoy.sh ultranest-fast-fixed4d 
# or to run real problems
# CMD bash runreal.sh ultranest-fast-fixed4d
# by default we run the testsampler:
CMD bash runtoy.sh testsampler && bash runreal.sh testsampler
ENV FERMI_DIR=/opt/conda/share/fermitools/
ENV PFILES=/opt/conda/share/fermitools/syspfiles/
