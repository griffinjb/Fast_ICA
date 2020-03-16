import numpy as np
import matplotlib.pyplot as plt
from NN_AA import liveplot
# np.random.seed(42)


# This code will explore Fast ICA as described by Aapo Hyvarinen in "Independent Component Analysis: Algorithms and Applications".

# Model: x = As :
#           x - observed mixtures
#           A - mixing matrix
#           s - pure random variables

# Consider two uniform distributions
# min/max - -1/1

def uniform_gen(min=-1,max=1):
    while 1:
        yield(np.random.uniform(-1,1))

def normal_gen():
    while 1:
        yield(np.random.normal(0,1))

def test1_square_to_parallelogram():

    s = [uniform_gen(),uniform_gen()]

    #A = np.random.uniform(1,2,[2,2])
    A = np.array([[2,3],[2,1]])

    xs = []

    for i in range(10000):
        ss = []
        for si in s:
            ss.append(si.__next__())
        sa = np.array(ss)[:,None]
        x = A@sa
        xs.append(x)

    xs = np.array(xs)[:,:,0]

    plt.hist2d(xs[:,0],xs[:,1],bins=100)
    plt.show()

# test1_square_to_parallelogram()

# Additive mixtures of random variables tend towards the gaussian distribution. Thus, non-gaussianity is a metric of "purity".

# Note: n-dimensional non-gaussian optimization has 2*n local maxima. Two for each independent component (-/+)

# Kurtosis : E{y^4}-3(E{y^2})^2
# Magnitude Maximization -> non-gaussianity
# Metric of Curvature or Spikiness

# Kurtosis is sensitive to outliers
# Requires multiple values for estimation of Expectation
# Not robust, do not use!

# entropy : H = -sum{Plog(P)}

# Gaussian variable has the largest entropy! (scaled by variance)

# Negentropy : J = H(gauss) - H(y)
# non-negative , invariant under linear transforms
# Maximization yields non-gaussianity

# "In some sense, the optimal estimator of nongaussianity"
# However, an estimate requires the PDF, so we seek simplification.

# Mutual Information : I(y1,..yn) = sum(H(yi)-H(y))
# Equivalent to Kullback-Leibler divergence between the joint density f(y) and the product of its marginal densities.

# non-negative, and zero iff independent

# Compares code length advantage of joint vs. marginals.
# Independence implies no advantage, as not MI

# I(y1,..,yn) = C-sum(J(yi))

# MI minimization <-> nongaussianity maximization
# Estimates constrained to be uncorrelated

# Maximum Likelihood : L = sum{sum{log(fi(wi xi))+Tlog|detW|}}
# W = A^-1
# fi -> density functions of si (not always available)

# Infomax <-> maximum likelhood estimation under conditions

# Projection Pursuit - somewhat a superset of ICA

# Preprocessing for ICA:

#Centering - 0 mean

def center_gen(dg):
    E = dg.__next__()
    N = 1
    for d in dg:
        E *= N
        E += d
        N += 1
        E /= N
        yield(E)

# Whitening
# linear transform s.t uncorrelated, unit variance
# E(x x.T) = I

def covariance_gen(dg,return_complete_C=False):
    if return_complete_C:
        x1 = dg.__next__()
        C = x1@x1.T
        N = 1
        ds = []
        for d in dg:
            ds.append(d)
            C *= N
            C += d@d.T
            N += 1
            C /= N
        for d in ds:
            yield((C,d))
    else:
        x1 = dg.__next__()
        C = x1@x1.T
        N = 1
        for d in dg:
            C *= N
            C += d@d.T
            N += 1
            C /= N
            yield((C,d))

def whitening_gen(dg):
    cg = covariance_gen(dg,1)
    for (C,d) in cg:
        S,V = np.linalg.eig(C)
        S[S!=0] = 1/np.sqrt(S[S!=0])
        S_ = S*np.eye(len(S))
        yield(V@S_@V.T@d)

def test_data_MD_sin(dim=3):
    t = 0
    dt = .01
    while 1:
        # yield(np.array([np.sin(w*t) for w in range(dim)])[:,None])
        dd = np.random.uniform(-1,1,[3,1])
        d2 = dd.copy()
        d2[0] += .3*dd[1]+.8*dd[2]
        yield(d2)
        t += dt

def array_gen(arr):
    for row in range(arr.shape[0]):
        yield(arr[row,:][:,None])

def test2_square_to_parallelogram_to_square():

    s = [uniform_gen(),uniform_gen()]

    #A = np.random.uniform(1,2,[2,2])
    A = np.array([[2,3],[2,1]])

    xs = []

    for i in range(10000):
        ss = []
        for si in s:
            ss.append(si.__next__())
        sa = np.array(ss)[:,None]
        x = A@sa
        xs.append(x)

    xs = np.array(xs)[:,:,0]

    ws = []
    wdg = whitening_gen(array_gen(xs))
    for d in wdg:
        ws.append(d)
    ws = np.array(ws)[:,:,0]

    plt.hist2d(ws[:,0],ws[:,1],bins=100)
    # plt.hist2d(xs[:,0],xs[:,1],bins=100)
    plt.show()

# Whitening yields A~ from A, A~ is orthogonal
# An orthogonal matrix has only n(n-1)/2 DOF

# PCA may be applied at this step

# test2_square_to_parallelogram_to_square()


#def pdf_estimate(dg):
 #gross big histogram. expensive to bin on the fly

# CDF is probability that val is less than. summation of pdf
def cdf_estimate(dg):
    # Alternate unit summation and update via * , ++ , /
    # sorting is the answer
    ds = []
    for d in dg:
        ds.append(d[0,0])
        ds.sort()
        # argmin distance function

def test_cdf_est():
    dg = array_gen(np.random.normal(0,1,[1000,1]))

    i = 0
    for d in cdf_estimate(dg):
        if i:
            plt.plot(d)
            plt.show(block=False)
            plt.pause(.01)
            plt.cla()
        i+=1

# Fast ICA for one UNIT

# Finds a direction that maximizes non_gauss via negentropy

# Basic Algorithm:

# - Choose initial weight vector w

# - let w+ = E{xg(w.T @ x)} - E{g'(w.T@x)}w   update

# - let w = w+ / ||w+||    unit norm weights

# - if not converged, 2

# use gram-schmidt decorrelation to prevent same convergence

# used for estimation of E
class expectation:

    def __init__(self,w=.0001):
        self.E = 0
        self.w = w
        self.unset = 1

    def update(self,x):
        if self.unset:
            self.E = x 
            self.unset = 0
        else:
            self.E = (1-self.w)*self.E+self.w*x

# duplicates generator output for splitting network
def multi_gen(dg,N):
    for d in dg:
        for i in range(N):
            yield(d)

# metric of non-gaussianity
def negentropy(x):
    return(-np.exp(-x**2/2))

# first derivative
def negentropy_d1(x):
    return(x*np.exp(-x**2/2))

# second derivative
def negentropy_d2(x) :
    return(np.exp(-x**2/2)*(x**2-1))

class whitening:

    def __init__(self,dg):
        self.dg = dg 

    def gen(self):
        cg = covariance_gen(self.dg)
        for (C,d) in cg:
            S,V = np.linalg.eig(C)
            S[S!=0] = 1/np.sqrt(S[S!=0])
            S_ = S*np.eye(len(S))
            
            self.inv_tf = V@S@V.T

            yield(V@S_@V.T@d)

class fast_ICA:

    def __init__(self,N_components=1,iswhite=False):
        self.iswhite = iswhite
        self.ws = []
        self.N_components = N_components
        self.ws_prev = []

    def gen(self,dg):

        g = negentropy_d1 
        gp = negentropy_d2

        W = whitening(dg)
        wdg = W.gen()

        Es = [[expectation(),expectation()] for i in range(self.N_components)]

        Ec = expectation(.001)

        for x in wdg:
            if 'list' in str(type(self.ws)):
                self.ws = np.random.normal(0,1,[self.N_components,len(x)])
                lens = np.linalg.norm(self.ws,axis=1)
                lens = lens * np.ones(self.ws.shape)
                self.ws = self.ws / lens

                self.ws_prev = np.zeros(self.ws.shape)

            converged = self.N_components - np.sum((self.ws@self.ws_prev.T)*np.eye(self.N_components))
            Ec.update(converged)

            # converged = np.abs(self.N_components-(np.abs(self.ws[0,:]@self.ws_prev.T[:,0])+np.abs(self.ws[1,:]@self.ws_prev.T[:,1])))

            if Ec.E > 1e-06 or 1:

                ws = self.ws.copy()
       
                for r in range(self.N_components):

                    w = self.ws[r,:]
        
                    # Expectation update
                    Es[r][0].update(x*g(w@x))
                    Es[r][1].update(gp(w@x))
        
                    # Newton iteration
                    w = Es[r][0].E - (Es[r][1].E*w)[:,None]

                    # unit norm 
                    w /= np.linalg.norm(w)
        
                    # decorrelation of N_components > 1
                    for i in range(r):
                        wi = ws[i,:][:,None]
                        w = w - w.T@wi*wi

                    w /= np.linalg.norm(w)

                    ws[r,:] = w[:,0]

                self.ws_prev = self.ws.copy()
        
                self.ws = ws
                # print(Ec.E)
            else:
                print('',end='')

            yield(self.ws@x)

    def transform(self,x):
        print('tmp')

def tf_gen(dg,A):
    for d in dg:
        yield(A@d)

def multi_gen_to_vect(dgs):
    v = np.zeros([len(dgs),1])
    while 1:
        i = 0
        for dg in dgs:
            v[i] = dg.__next__()
            i += 1
        yield(v)

def live_hist(X,name='',bins=100):
    plt.figure(name)
    plt.cla()
    plt.hist(X,bins=bins)
    plt.plot(block=False)
    plt.pause(.001)

def live_hist2d(X,bins=100,name='',xlabel='',ylabel='',save=0):

    plt.figure(name)
    plt.cla()
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist2d(X[:,0],X[:,1],bins=bins)
    plt.plot(block=False)
    plt.pause(.001)

    if save:
        

def live_quiver(X,name='quiv'):
    
    origin = [0],[0]

    plt.figure(name)
    plt.cla()
    plt.quiver(*origin,X[:,0],X[:,1])
    plt.plot(block=False)
    plt.pause(.001)    

def triangular_gen():
    while 1:
        yield(np.random.triangular(-1,0,1))

def pareto_gen():
    while 1:
        yield(np.random.pareto(300))

def zero_mean_gen(dg):
    E = expectation()
    for d in dg:
        E.update(d)
        yield(d-E.E)

def test_fica():

    # works
    dg = multi_gen_to_vect([uniform_gen(),uniform_gen()])

    # works
    # dg = multi_gen_to_vect([triangular_gen(),triangular_gen()])

    # doesn't seem to work
    # dg = multi_gen_to_vect([zero_mean_gen(pareto_gen()),zero_mean_gen(pareto_gen())])

    # works, uses info from uniform.
    # dg = multi_gen_to_vect([zero_mean_gen(pareto_gen()),uniform_gen()])

    # Does not work due to radial symmetry of joint dist.
    # dg = multi_gen_to_vect([normal_gen(),normal_gen()])

    A = np.array([[2,3],[2,1]])
    # A = np.eye(2)

    dg = tf_gen(dg,A)

    ICA = fast_ICA(N_components=2)
    W = whitening(dg)

    wdg = W.gen()

    idg = ICA.gen(wdg)

    ws = []
    ctr = 0 
    for d in dg:
        ws.append(d[:,0])
        # live_hist(np.array(ws),'idg_hist')
        ctr += 1
        if not ctr%1000 and ctr > 5:
            live_hist2d(np.array(ws),'idg_hist')
            # liveplot(ICA.ws.T)
            # live_quiver(np.array(ICA.ws),'quiv')

        if len(ws) > 10000:
            ws = ws[-10000:]

def test_plot():
    # works
    dg = multi_gen_to_vect([uniform_gen(),uniform_gen()])

    # works
    # dg = multi_gen_to_vect([triangular_gen(),triangular_gen()])

    # doesn't seem to work
    # dg = multi_gen_to_vect([zero_mean_gen(pareto_gen()),zero_mean_gen(pareto_gen())])

    # works, uses info from uniform.
    # dg = multi_gen_to_vect([zero_mean_gen(pareto_gen()),uniform_gen()])

    # Does not work due to radial symmetry of joint dist.
    # dg = multi_gen_to_vect([normal_gen(),normal_gen()])

    # lbls= ['Original Joint Distribution','X_0','X_1']
    # lbls= ['Mixture Joint Distribution','X_0','X_1']
    # lbls= ['Whitened Joint Distribution','X_0','X_1']
    lbls= ['Post FastICA Joint Distribution','X_0','X_1']

    A = np.array([[2,3],[2,1]])
    # A = np.eye(2)

    dg = tf_gen(dg,A)

    ICA = fast_ICA(N_components=2)
    W = whitening(dg)

    wdg = W.gen()

    idg = ICA.gen(wdg)

    ws = []
    ctr = 0 
    for d in idg:
        ws.append(d[:,0])
        # live_hist(np.array(ws),'idg_hist')
        ctr += 1
        if not ctr%1000 and ctr > 5:
            live_hist2d(np.array(ws),100,lbls[0],lbls[1],lbls[2])
            # liveplot(ICA.ws.T)
            # live_quiver(np.array(ICA.ws),'quiv')

        if len(ws) > 10000:
            ws = ws[-10000:]

def test_prior():

    # estimate the CDF,PDF

    # Pursue this distribution 

    # What is a good cost funtion?

    print('')

def test_audio():
    print('')        

def test_eeg():
    print('')

def test_nonlinear_mixture():
    print('')









# Ah! yes! neural ICA via degaussianity
# Linear combination is oh yah

# so the joint density of X0, X1
# 





























# you need a grid with selective activations. If you have one good direction you have many.


# spatial symmetry with vortex of sensors

# peak rotational autocorrelation for detecting symmetries (angular correlations)

# Use spiral to improve resolution by 













