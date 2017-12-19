#!/usr/local/bin/python
""" This script iterates the following two steps for a fixed length of time:
* Sample a pair (n,k) from the prior
* Sample k given n from the posterior
"""

from CFTP import *
from optparse import OptionParser
import os
import errno


parser = OptionParser()
parser.add_option("-r", "--numSeqs",
                  type="int", dest="R", default=2,
                  help="Number of sequences")
parser.add_option("-s", "--samplesPerSeq",
                  type="int", dest="samplesPerSeq", default=10,
                  help="Number of samples per sequence")
parser.add_option("-b", "--beta",
                  type="float", dest="Beta", default=0,
                  help="Beta for lower level of hierarchy")
parser.add_option("-B", "--beta0",
                  type="float", dest="Beta0", default=0,
                  help="Beta for upper level of hierarchy")
parser.add_option("-t", "--theta",
                  type="float", dest="theta", default=1,
                  help="Theta for lower level of hierarchy")
parser.add_option("-T", "--theta0",
                  type="float", dest="theta0", default=1,
                  help="Theta for upper level of hierarchy")
parser.add_option("-l", "--limit",
                  type="float", dest="limit", default=3,
                  help="Time limit in hours")

(options, args) = parser.parse_args()

# Memoize triangular array
logf = GetArray(options.samplesPerSeq,options.Beta)
print("Done computing triangular array")

# Create path to save results in
path = "r%i_s%i_b%.2f_B%.2f_t%.2f_T%.2f"%(options.R,options.samplesPerSeq,options.Beta,
                                          options.Beta0,options.theta,options.theta0)
try:
    os.makedirs(path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise


# Start a time counter
signal.alarm(int(options.limit*3600))
kPost = 1
ind=1

try:
    while not kPost=="NoOutput":
        # Simulate some data from the prior
        u,l = SampleCRF([options.samplesPerSeq]*options.R,options.theta,
                         options.Beta,options.theta0,options.Beta0)
        # Array of counts
        n = np.array([ [l[r].n[i] if i in l[r].n else 0 for i in u.n.keys()] for r in range(options.R)])
        # Array of counts from auxiliary sequence
        k = np.array([ [l[r].k[i] if i in l[r].k else 0 for i in u.n.keys()] for r in range(options.R)])
        # Save n and k sampled from prior
        np.savetxt(path+"/n%i.txt"%ind,n,fmt="%i")
        np.savetxt(path+"/k%i_prior.txt"%ind,k,fmt="%i")
        # Sample posterior
        kPost, stepsToCoalescence = SamplePosteriorUntilSuccess(n,options.theta,options.Beta,
                                                                options.theta0,
                                                                options.Beta0,logf)
        if kPost=="NoOutput":
            pass
        else:
            np.savetxt(path+"/k%i_post.txt"%ind,kPost,fmt="%i")
        outFile = open(path+"/steps%i.txt"%ind,"w")
	outFile.write(str(stepsToCoalescence))
	outFile.close()
        ind += 1
except TimeoutException:
    pass


