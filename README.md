# Posterior sampling algorithms for the Hierarchical Pitman-Yor Process

The module `CFTP.py` implements the following three algorithms for posterior inference in the Hierarchical Pitman-Yor Process:

1. A blocked Gibbs sampler
2. A perfect sampler based on coupling from the past. 
3. A method for computing unbiased posterior moments based on Markov chain couplings [1].

The three algorithms are described in [2], and the notation used in the code is based on that paper. Please cite [2] if you find the module useful. 

[1] Glynn, P. W. and C.-h. Rhee (2014). Exact estimation for markov chain equilibrium expectations. *Journal of Applied Probability* 51(A), 377–389.

[2] Bacallado, S., Power, S., Favaro, S., and Trippa, L. (2017). Perfect Sampling of the Posterior in the Hierarchical Pitman–Yor Process. 

## Example script

The script `SimulateForFixedTime.py` may be used to estimate the running time of the perfect sampler in a well-specified model, under a fixed setting of the parameters.

## Dataset

The contingency table used as an example in Section 4 of [2] is made available as `example.csv`.

