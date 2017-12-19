""" Copyright 2017 Sergio Bacallado.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

#########
# Local imports
#
import signal
import numpy as np
from numpy.random import rand,choice,gamma,dirichlet
from scipy.stats import zipf,geom
#########

#########
# Define a mechanism to stop code after a set time limit, by calling
#
# signal.alarm(timeLimit)
#
# When timeLimit seconds have elapsed after executing this command, a TimeoutException is raised.
########
class TimeoutException(Exception):              # Custom exception class
    pass
def timeout_handler(signum, frame):             # Custom signal handler
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)  # Change the behavior of SIGALRM
##############


class TwoParamUrn:
    """ This class represents the urn in a Chinese Restaurant Process.

    """
    def __init__(self,theta,Beta,base=None):
        self.base = base
        self.theta = theta
        self.beta = Beta
        self.weights = {}
        self.black = theta
        self.total = theta
        self.n = {}
        self.k = {}

    def Sample(self):
        if rand() < self.black/self.total:
            # Draw a species from the base
            i = self.SampleBase()
            self.black += self.beta
            if i in self.weights:
                self.weights[i] += (1-self.beta)
            else:
                self.n[i] = 0
                self.weights[i] = (1-self.beta)
        else:
            i = choice(self.weights.keys(),p=np.array(self.weights.values())/(self.total-self.black))
            self.weights[i] += 1
        self.n[i] += 1
        self.total += 1
        return i

    def SampleBase(self):
        if self.base==None:
            return max(self.n.keys())+1 if len(self.n)>0 else 0
        else:
            i = self.base.Sample()
            if i in self.k:
                self.k[i] += 1
            else:
                self.k[i] = 1
            return i


def SampleCRF(numSamples,theta,Beta,theta0,Beta0):
    """ Outputs a number of sequences of species of lengths specified by numSamples, drawn from the
        Chinese Restaurant Franchise. The second output is an array of sequence by species with the
        number of times a species was drawn from upper level of the hierarchy.

    """
    upper = TwoParamUrn(theta0,Beta0)
    lower = []
    for r in range(len(numSamples)):
        lower.append(TwoParamUrn(theta,Beta,base=upper))
        for i in range(numSamples[r]):
            lower[r].Sample()

    return upper,lower


def MissingMasses(theta,Beta,theta0,Beta0,n,k,sampleIndex):
    """ Returns the missing-in-sample mass and the missing mass estimators given n, k
        for one of the populations in the dataset, indicated by 'sampleIndex'.

    """
    kr = k.sum(1)
    nr = n.sum(1)

    # Update weights of urns
    lowerBlack = theta+Beta*kr[sampleIndex]
    lowerTotal = theta+nr[sampleIndex]
    upperBlack = theta0 + k.shape[1]*Beta0
    upperTotal = theta0 + kr.sum()

    # Return missing-in-sample and missing mass
    return lowerBlack/lowerTotal, lowerBlack/lowerTotal*upperBlack/upperTotal


def SampleNewSpeciesGivenUrn(numSamples,theta,Beta,theta0,Beta0,n,k):
    """ This method is equivalent to calling SampleCRFGivenUrn with the same inputs
        but outputing only the number of new species discovered

    """
    kr = k.sum(1)
    nr = n.sum(1)

    lowerBlack = {}
    lowerTotal = {}
    # Update weights of urns
    for r in range(n.shape[0]):
        lowerBlack[r] = theta+Beta*kr[r]
        lowerTotal[r] = theta+nr[r]
    upperBlack = theta0 + k.shape[1]*Beta0
    upperTotal = theta0 + kr.sum()

    newTables = 0
    for r in range(n.shape[0]):
        newTables += SampleNewTables(numSamples[r],lowerBlack[r],lowerTotal[r],Beta)

    return SampleNewTables(newTables,upperBlack,upperTotal,Beta0)

def SampleNewTables(numSamples,black,total,beta):
    """ Samples the number of new tables observed when numSamples samples are drawn from a Chinese
        Restaurant Process with initial conditions determined by (black, total), and parameter beta.

    """
    n = 0
    for i in range(numSamples):
        if rand() < black/total:
            n+=1
            black+=beta
        total+=1
    return n

def SampleCRFGivenUrn(numSamples,theta,Beta,theta0,Beta0,n,k):
    """ Outputs a number of sequences of species of lengths specified by numSamples, drawn from the CRF
        initialised with the sufficient statistics n and k. The inputs n,k cannot have columns equal
        to 0.

    """
    upper = TwoParamUrn(theta0,Beta0)
    lower = []
    for r in range(len(numSamples)):
        lower.append(TwoParamUrn(theta,Beta,base=upper))

    kr = k.sum(1)
    ki = k.sum(0)
    nr = n.sum(1)

    # Update weights of urns
    for r in range(len(numSamples)):
        lower[r].weights = {i: n[r,i]-k[r,i]*Beta for i in range(n.shape[1])}
        lower[r].black = theta+Beta*kr[r]
        lower[r].total = theta+nr[r]
        lower[r].n = {i: n[r,i] for i in range(n.shape[1])}
        lower[r].k = {i: k[r,i] for i in range(k.shape[1])}

    upper.weights = {i: ki[i]-Beta0 for i in range(k.shape[1])}
    upper.black = theta0 + k.shape[1]*Beta0
    upper.total = theta0 + ki.sum()
    upper.n = {i: ki[i] for i in range(k.shape[1]) }

    for r in range(len(numSamples)):
        for i in range(numSamples[r]):
            lower[r].Sample()

    return upper, lower


def GetArray(maxn,Beta):
    """ Computes a triangular array of numbers required for posterior sampling in the hierarchical
        Pitman-Yor process.

    """
    logf = np.zeros([maxn+1,maxn+1])
    for n in range(1,maxn+1):
        for k in range(1,n+1):
            if n==k:
                logf[1,1] = 0
            else:
                a = np.array([np.log(n-k-1+ell*(1-Beta)) + logf[n-k+ell-1,ell] for ell in range(1,k+1)])
                c = max(a)
                logf[n,k] = c + np.log(np.exp(a-c).sum())
    return logf


def GibbsSampler(n,k,nSteps,theta,Beta,theta0,Beta0,logf=None):
    """ Simulates the Gibbs sampler for a fixed number of steps, and outputs its final state.

    """
    R,I = np.shape(n)
    G = np.ones(R)
    D = np.ones(I+1)/(I+1.0)
    try:
        logf.size
    except AttributeError:
        logf = GetArray(n.max(),Beta)

    for step in range(nSteps):
        kR = k.sum(1)
        kI = k.sum(0)
        # Resample G
        for r in range(R):
            G[r] = gamma(theta/Beta+kR[r])
        # Resample D
        param = kI + 1-Beta0 -1
        param = np.append(param,theta0+I*Beta0)
        D = dirichlet(param)[:-1]
        # Resample k
        for i in range(I):
            for r in range(R):
                if n[r,i] == 0:
                    k[r,i] = 0
                elif n[r,i] == 1:
                    k[r,i] = 1
                else:
                    x = [logf[n[r,i],kri] + kri*np.log(G[r]*D[i]*Beta) for  kri in range(1,n[r,i]+1)]
                    probs = np.exp(x-max(x))  # Avoid numerical overflow
                    probs = probs/sum(probs)
                    k[r,i] = choice(range(1,n[r,i]+1),p=probs)
    return G,D,k


## Coupling from the past

def SamplePosterior(n,theta,Beta,theta0,Beta0,logf=None,firstH=None):
    """ This function outputs either a sample from the posterior of k, or a failure message.

    """
    try:
        logf.size
    except AttributeError:
        logf = GetArray(n.max(),Beta)

    lowH = (n>0).sum()
    highH = n.sum()
    if firstH==None:
        H = lowH
    else:
        H = firstH
    R,I = n.shape
    nRMax = int(n.sum(1).max())
    nIMax = int(n.sum(0).max())
    U = []
    g = []
    d = []
    E = []
    Ep = []
    Es = []
    e0 = gamma(theta0)

    stepsList = []
    while True:
        if len(Es)<H:
            Es.extend(gamma(1,size=int(H-len(Es))))
        a = sum(Es[:int(H)])+e0
        stepsSampled = 1
        while True:
            # Add to randomness sources if necessary
            while len(U)<stepsSampled:
                U.append(rand(R,I))
                g.append(gamma(theta/Beta,size=R))
                d.append(gamma(1-Beta0,size=I))
                E.append(gamma(1,size=[I,nIMax]))
                Ep.append(gamma(1,size=[R,nRMax]))
            print len(U), stepsSampled
            kL = (n>0).astype(int)
            kU = n.copy()
            for x in range(stepsSampled):
                m = stepsSampled-x-1
                # Propagate upper limit with fixed a
                TakeStep(n,kU,U[m],g[m],d[m],E[m],Ep[m],a,Beta,logf)
                # Propagate lower limit with fixed a
                TakeStep(n,kL,U[m],g[m],d[m],E[m],Ep[m],a,Beta,logf)
            # If converged return the limit of k
            if (kU==kL).all():
                k = kU
                stepsList.append(stepsSampled)
                break
            stepsSampled *= 2

        print("H: %i, sum of k: %i, steps to coalescence:%i"%(H,k.sum(),stepsSampled))
        if k.sum() == H:
            print("Success!")
            return True,k,stepsList
        elif k.sum() < H:
            # Overshot the potential solution, reduce H
            highH = H
            H = H-max(1,np.floor((H-max(lowH,k.sum()))/2))
            # If this H is smaller than the largest H that undershoots the solution, quit
            if H <= lowH:
                print "Failed realisation"
                return False,H,stepsList
        else:
            # Undershot the potential solution, increase H
            lowH = H
            H = H+max(1,np.floor((min(highH,k.sum())-H)/2))
            # If this H is higher than the largest H that undershoots the solution, quit
            if H >= highH:
                print "Failed realisation"
                return False,H,stepsList


def TakeStep(n,k,U,g,d,E,Ep,a,Beta,logf):
    """ Takes a step of the coupled Markov chains given sources of randomness.

    """
    R,I = n.shape
    kI = k.sum(0).astype(int)
    kR = k.sum(1).astype(int)
    D = d+np.array([E[i,:kI[i]-1].sum() for i in range(I)])
    G = g+np.array([Ep[r,:kR[r]].sum() for r in range(R)])
    kold = k.copy() # new check
    for r in range(R):
        for i in range(I):
            if n[r,i] in [0,1]:
                k[r,i] = n[r,i]
            else:
                x = [logf[n[r,i],kri] + kri*np.log(G[r]*D[i]*Beta/a) for  kri in range(1,n[r,i]+1)]
                probs = np.exp(x-max(x))
                probs = probs/sum(probs)
                k[r,i] = len(probs) - sum(U[r,i] < np.cumsum(probs)) + 1


def SamplePosteriorUntilSuccess(n,theta,Beta,theta0,Beta0,logf=None):
    """ This function will apply the exact sampler until the integer equation is
    satisfied. If a TimeoutException is raised before termination, we just return
    runtime statistics.

    Outputs:
    - out: Outputs n,k sampled. If the algorithm did not terminate, equals "NoOutput".
    - stepsToCoalescence: a list of steps to coalescence for the coupling, for successive
      values of H used. The number of elements of this list is the number of H attempted.
    """

    stepsToCoalescence = []
    firstH = None
    try:
        while True:
            success,out,steps = SamplePosterior(n,theta,Beta,theta0,Beta0,logf,firstH=firstH)
            stepsToCoalescence.append(steps)
            if not success:
                firstH = out
                continue
            else:
                return out,stepsToCoalescence
                break
    except TimeoutException:
        return "NoOutput",stepsToCoalescence


def funcMissingMasses(n,k,theta,Beta,theta0,Beta0):
    a,b = MissingMasses(theta,Beta,theta0,Beta0,n,k,1)
    return np.array([a,b])

def UnbiasedEstimate(n,k,theta,Beta,theta0,Beta0,func,minT=1000,logf=None,zipfParam=1.5):
    """ This function simulates an unbiased estimate following the algorithm of Rhee and Glynn.

    func is a function that accepts n,k,theta,Beta,theta0,Beta0 and outputs a number or a numpy array
    """
    # Draw the length of the Markov chain from a power law
    #T = minT + np.random.zipf(a=zipfParam)
    # Draw the length of the Markov chain from a geometric distribution
    T = minT + np.random.geometric(zipfParam)
    print("The number of steps in the Markov chain is %i"%T)

    # Initialize variables
    R,I = np.shape(n)
    G1 = np.ones(R)
    G2 = np.ones(R)
    try:
        logf.size
    except AttributeError:
        logf = GetArray(n.max(),Beta)

    est = func(n,k,theta,Beta,theta0,Beta0)
    k1 = k.copy()   # This is the equivalent of k
    k2 = k.copy()   # This is the equivalent of \tilde k
    for step in range(1,T+1):
        kR1 = k1.sum(1)
        kI1 = k1.sum(0)
        kR2 = k2.sum(1)
        kI2 = k2.sum(0)
        # Resample G
        for r in range(R):
            auxGammas = gamma(1,size=max(kR1[r],kR2[r]))
            G1[r] = gamma(theta/Beta)
            G2[r] = G1[r]
            G1[r] += sum(auxGammas[:kR1[r]])
            G2[r] += sum(auxGammas[:kR2[r]])
        # Resample D
        auxGammas = gamma(1,size=[I,max(kI1.max(),kI2.max())])
        auxGammas2 = gamma(1-Beta0,size=I)
        auxGamma = gamma(theta0+I*Beta0)
        D1 = auxGammas2 + np.array([sum(auxGammas[i,:kI1[i]-1]) for i in range(I)])
        D1 = D1/(D1.sum()+auxGamma)
        D2 = auxGammas2 + np.array([sum(auxGammas[i,:kI2[i]-1]) for i in range(I)])
        D2 = D2/(D2.sum()+auxGamma)
        # Resample k
        unif = np.random.uniform(size=k.shape)
        UpdateK(k1,n,I,R,G1,D1,unif,logf,Beta)
        if step>1:
            UpdateK(k2,n,I,R,G2,D2,unif,logf,Beta)
        # Terminate if coupling has merged
        if (k1==k2).all():
            break
        # Otherwise continue sum
        #denom = (1-zipf.cdf(step-minT-1,a=zipfParam)) if step>minT else 1.0
        denom = (1-geom.cdf(step-minT-1,p=zipfParam)) if step>minT else 1.0
        summand = (func(n,k1,theta,Beta,theta0,Beta0)-func(n,k2,theta,Beta,theta0,Beta0))/denom
        est += summand
        print summand
    print est
    return est

def UpdateK(k,n,I,R,G,D,unif,logf,Beta):
    for i in range(I):
        for r in range(R):
            if n[r,i] == 0:
                k[r,i] = 0
            elif n[r,i] == 1:
                k[r,i] = 1
            else:
                x = [logf[n[r,i],kri] + kri*np.log(G[r]*D[i]*Beta) for  kri in range(1,n[r,i]+1)]
                probs = np.exp(x-max(x))  # Avoid numerical overflow
                probs = probs/sum(probs)
                k[r,i] = sum(unif[r,i]>np.cumsum(probs))+1
            if k[r,i]>n[r,i]:
                print("shout")


