Title
------

Rapid and robust Bayes factor computation for Doppler shift exoplanet discovery

Data
-----

* from challenge I
* from challenge II

Correctness diagnostic: no additional planets found

Model
-----

 * RV
   * with period ordering
 * ExpSquaredKernel * (Exp-sine-squared kernel + Exp-sine-squared kernel)
 * RV prior as in challenge
 * GP prior:
   * uninformative priors
   * sine period1: biased to 1 year (or period of star)
   * sine period2: biased to 1 day (or rotation period of star)

Method
------

 * Computation
   * juliet
   * ultranest with step sampler
 * Speed-ups:
   * first hot start:
     * guess offset from data mean and (max-min)/2. Use student-t
     * get first period from data with Lomb-scargle periodogram. add 5% plateau.
     * get jitter centered around data error bar +- (max-min)/2 of values
   * run on 2-planets first
   * use trained GP posterior as hot start for other runs
 * Robustness:
   * Leave out 1/4 of the data --> parameters must be stable
   * parametric bootstrap: repeat 40 times:
     * generate data with 0, 1 planets 
     * compute BF distribution

Results
-------

 * Show speed-up of first hot start
 * Show speed-up of further runs (1, 0 planets based on 2 planets)
 * Show unbiasedness of hot start runs (delta-lnZ)
 * Show BF robustness compared to MultiNest runs
   * scatter of multiple runs, same dataset as Nelson+
 * Show false positive rate based on challenge
 * Show model diagnostic predicting left-out data

Discussion:
 * We attempt to provide a robust modelling workflow
