import numpy as np 
import matplotlib.pyplot as plt 
from bisect import insort, bisect
from scipy.io import wavfile
from random import shuffle

def sin_gen(w=1,A=1,phi0=0):
	t = 0
	dt = .1
	while 1:
		yield(A*np.sin(w*t+phi0))
		t+=dt

def uniform_gen():
	while 1:
		yield(np.random.uniform(0,1))

def normal_gen():
	while 1:
		yield(np.random.normal(0,1))

def sum_gen(gs):
	while 1:
		s = 0
		for g in gs:
			s += g.__next__()
		yield(s)

def wav_gen(fn='t1o.wav',shuffle_en=False):
	d = wavfile.read(fn)[1].copy()
	if shuffle_en:
		shuffle(d)
	# while 1:
	for sample in d:
		yield(sample)

def gexp(d,std,mu=0):
	return(np.exp(-.5*(d/std)**2))

# it needs to be an attractor for a whole region - dist weight
# gaussian attractor
# abuse symmetry 
class Distribution:

	def __init__(self,N_bins,interp='linear'):
		self._ = 0
		self.N_bins = N_bins

		# self.CDF = [_ for _ in np.linspace(0,1,N_bins)]
		self.CDF = []
		self.interp = interp


	def update(self,x):
		symm_w = 0.01
		attractor_w = 0
		smooth_w = .05
		rate_w = 1

		if len(self.CDF) < self.N_bins:
			self.CDF += [x]
			self.CDF.sort()
		else:
			idx = bisect(self.CDF,x)
			if idx == len(self.CDF):
				self.CDF[idx-1] += x
				self.CDF[idx-1] /= 2
			else:
				mod = 0
				if abs(self.CDF[idx-1]-x) < abs(self.CDF[idx]-x):
					mod = 1
				val = self.CDF[idx-mod]

				self.CDF[idx-mod] = (1-rate_w)*val + rate_w*x

				# Attractor - stability problems
				for i in range(len(self.CDF)):
					if i != idx-mod:
						d = i-(idx-mod)
						d /= self.N_bins
						w = gexp(d,.05)
						self.CDF[i] = (1-attractor_w)*self.CDF[i] + attractor_w*(self.CDF[i]*(1-w)+x*w)

				# Smoothing
				CDF_t = self.CDF.copy()
				for i in range(1,len(self.CDF)-1):
					self.CDF[i] = (1-smooth_w)*self.CDF[i] + (smooth_w)*(.5*CDF_t[i-1]+.5*CDF_t[i+1])

				# Symetry constraint
				def symm_mix(X,symm_w):
					X = np.array(X)
					X_inv = np.flip(min(X)+max(X)-X)
					X0 = X.copy()
					X0 = (1-symm_w)*X + symm_w*X_inv
					return(X0)

				self.CDF = symm_mix(self.CDF,symm_w)


	def map(self,x):
		if self.interp == 'step':
			return(np.argmin(np.abs(np.array(self.CDF)-x))/len(self.CDF))
		if self.interp == 'linear':
			nearest = np.argmin(np.abs(np.array(self.CDF)-x))
			if self.CDF[nearest] <= x:
				lower = nearest
				if lower == len(self.CDF)-1:
					upper = lower
					return(1)
				else:
					upper = lower + 1
			else:
				upper = nearest
				if upper == 0:
					lower = 0
					return(0)
				else:
					lower = upper - 1

			w = (x-self.CDF[lower])/(self.CDF[upper]-self.CDF[lower])

			return(((1-w)*lower + w*upper)/len(self.CDF))


	def imap(self,x):
		if self.interp == 'step':
			x *= len(self.CDF)-1
			x = int(np.round(x))
			return(self.CDF[x])
		if self.interp == 'linear':
			x *= len(self.CDF)-1
			w = x%1
			# floating point error
			if x > len(self.CDF)-1:
				x = len(self.CDF)-1
			return((1-w)*self.CDF[int(np.floor(x))] + w*self.CDF[int(np.ceil(x))])

	def nmap(self,x):
		if self.interp == 'step':
			lin_map = np.linspace(min(self.CDF),max(self.CDF),len(self.CDF))
			return(np.argmin(np.abs(np.array(lin_map)-x))/len(lin_map))
		if self.interp == 'linear':
			cmax = max(self.CDF)
			cmin = min(self.CDF)
			if cmax == cmin:
				return(cmin)
			w = (x-cmin)/(cmax-cmin)
			return(w)
			# return((1-w)*cmin+w*cmax)

def test1():

	D = Distribution(100)
	D2 = Distribution(10)
	
	
	dg = sum_gen([sin_gen(1,1),sin_gen(10,.4)])
	# dg = sum_gen([sin_gen(),normal_gen()])
	# dg = sin_gen()
	# dg = uniform_gen()
	# dg = normal_gen()
	
	dbuf = np.zeros(1000)
	
	for d in dg:
		D.update(d)
		D2.update(D.map(d))
	
		plt.figure('CDF')
		plt.cla()
		plt.plot(D.CDF)
		plt.show(block=False)
	
		plt.figure('2CDF')
		plt.cla()
		plt.plot(D2.CDF)
		plt.show(block=False)
	
		dbuf[:-1] = dbuf[1:]
		dbuf[-1] = D.map(d)	
		plt.figure('mapped')
		plt.cla()
		plt.plot(dbuf)
		plt.show(block=False)
		plt.pause(.01)

def liveplot(X,name=''):
	plt.figure(name)
	plt.cla()
	plt.plot(X)
	# plt.show(block=False)
	# plt.pause(.01)

def test2():
	
	D = Distribution(100)
	
	# dg_train = normal_gen()
	# dg_test = normal_gen()
	dg_test = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])
	dg_train = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])

	dbuf = np.zeros([100])
	db2 = np.zeros(100)
	
	for dtr,dte in zip(dg_train,dg_test):
		D.update(dtr)
		dbuf[:-1] = dbuf[1:]
		dbuf[-1] = D.map(dte)	
		db2[:-1] = db2[1:]
		db2[-1] = dte	
		liveplot(dbuf,'dbuf')
		liveplot(D.CDF,'cdf')
		liveplot(db2,'db2')

def test3():
	D = Distribution(100)
	
	dg_train = wav_gen()
	dg_test = wav_gen()	
	dbuf = np.zeros([100])
	db2 = np.zeros(100)
	
	for dtr,dte in zip(dg_train,dg_test):
		D.update(dtr)
		dbuf[:-1] = dbuf[1:]
		dbuf[-1] = D.map(dte)	
		db2[:-1] = db2[1:]
		db2[-1] = dte	
		liveplot(dbuf,'dbuf')
		liveplot(D.CDF,'cdf')
		liveplot(db2,'db2')

def test4():
	
	D = Distribution(100)
	D2 = Distribution(100)
	
	dg_train = normal_gen()
	# dg_test = normal_gen()
	# dg_test = sin_gen()
	dg_test = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])
	# dg_train = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])

	dbuf = np.zeros([100])
	db2 = np.zeros(100)
	
	for dtr,dte in zip(dg_train,dg_test):
		D.update(dte)
		D2.update(dtr)
		dbuf[:-1] = dbuf[1:]
		# dbuf[-1] = D.map(dte)
		dbuf[-1] = D.imap(D.nmap(dte))
		# dbuf[-1] = D.nmap(dte)
		db2[:-1] = db2[1:]
		db2[-1] = dte
		liveplot(dbuf,'dbuf')
		liveplot(D.CDF,'cdf_sin')
		liveplot(D2.CDF,'cdf_normal')
		liveplot(db2,'db2')
		plt.show(block=False)
		plt.pause(.01)

# to file

def test5():
	D = Distribution(100)
	D2 = Distribution(100)

	dg_train = wav_gen(shuffle_en=True)
	# dg_test = normal_gen()
	# dg_test = sin_gen()
	# dg_test = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])
	dg_test = wav_gen()
	# dg_train = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])

	dbuf = np.zeros([100])
	db2 = np.zeros(100)

	dout = .5*np.ones(44100*170)
	idx = 0
		
	for dtr,dte in zip(dg_train,dg_test):
		# D.update(dte)
		D2.update(dtr)

		dout[idx] = D2.map(dte)

		idx += 1

	dout -= .5
	dout *= np.iinfo(np.int16).max

	# dout *= (32767+32768-100)
	# dout -= 32766
	dout = dout.astype('int16')
	wavfile.write('test_1.wav',44100,dout)

def test6():
	D = Distribution(100)
	D2 = Distribution(100)
	
	dg_train = wav_gen(shuffle_en=True)
	dg_itrain = normal_gen()
	# dg_test = normal_gen()
	# dg_test = sin_gen()
	# dg_test = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])
	dg_test = wav_gen()
	# dg_train = sum_gen([sin_gen(),sin_gen(w=3,A=.3)])
	
	dbuf = np.zeros([100])
	db2 = np.zeros(100)
	
	# dout = .5*np.ones(44100*170)
	dout = np.zeros(44100*170)
	idx = 0
		
	for dtr,dte in zip(dg_train,dg_test):

		if idx <= 44100*3:
			# D.update(dte)
			D2.update(dtr)
			# D.update(dgi)

		dout[idx] = D2.imap(D2.nmap(D2.imap(D2.nmap(dte))))

		idx += 1

		if idx >= 44100*10:
			break
		
		dout -= dout.min()
		dout /= dout.max()
		
		dout -= .5
		dout *= np.iinfo(np.int16).max
		
		# dout *= (32767+32768-100)
		# dout -= 32766
		dout = dout.astype('int16')
		wavfile.write('test_xtra.wav',44100,dout)

	# dbuf[:-1] = dbuf[1:]
	# dbuf[-1] = D2.map(dte)
	# dbuf[-1] = D.imap(D.nmap(dte))
	# dbuf[-1] = D.nmap(dte)
	# db2[:-1] = db2[1:]
	# db2[-1] = dte



	# liveplot(dbuf,'dbuf')
	# liveplot(D.CDF,'cdf_sin')
	# liveplot(D2.CDF,'cdf_normal')
	# liveplot(db2,'db2')
	# plt.show(block=False)
	# plt.pause(.01)
	
# test6()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	







# old code


class Distribution_bad:

	def __init__(self,N_bins):
		self._ = 0
		self.N_bins = N_bins
		# [centroid,value]
		self.PDF = np.zeros([N_bins,2])
		self.CDF = np.zeros([N_bins,2])
		self.PDF[:,0] = np.linspace(0,1,N_bins)
		self.CDF[:,0] = np.linspace(0,1,N_bins)

	def update(self,x):
		centroid_step_size = 0
		distribution_step_size = .01


		centroid_index = np.argmin(np.abs(self.PDF[:,0]-x))

		dist = (x-self.PDF[centroid_index,1])
		if dist:
			self.PDF[centroid_index,0] += centroid_step_size * np.sign(dist)*dist

		self.PDF[centroid_index,1] += distribution_step_size

		self.PDF[:,1] /= np.sum(self.PDF[:,1])


class Distribution_old:

	def __init__(self,N_bins):
		self._ = 0
		self.N_bins = N_bins

		# self.CDF = [_ for _ in np.linspace(0,1,N_bins)]
		self.CDF = []


	def update(self,x):
		if len(self.CDF) < self.N_bins:
			self.CDF += [x]
			self.CDF.sort()
		else:
			idx = bisect(self.CDF,x)
			if idx == len(self.CDF):
				self.CDF[idx-1] += x
				self.CDF[idx-1] /= 2
			else:
				mod = 0
				if abs(self.CDF[idx-1]-x) < abs(self.CDF[idx]-x):
					mod = 1
				self.CDF[idx-mod] += x
				self.CDF[idx-mod] /= 2

		self.PDF = 1/np.diff(self.CDF)


	def map(self,x):
		return(np.argmin(np.abs(np.array(self.CDF)-x))/len(self.CDF))


class Distribution_old_but_not_as_old:

	def __init__(self,N_bins):
		self._ = 0
		self.N_bins = N_bins

		# self.CDF = [_ for _ in np.linspace(0,1,N_bins)]
		self.CDF = []


	def update(self,x):
		symm_w = 0.01
		attractor_w = 0
		smooth_w = .05
		rate_w = 1

		if len(self.CDF) < self.N_bins:
			self.CDF += [x]
			self.CDF.sort()
		else:
			idx = bisect(self.CDF,x)
			if idx == len(self.CDF):
				self.CDF[idx-1] += x
				self.CDF[idx-1] /= 2
			else:
				mod = 0
				if abs(self.CDF[idx-1]-x) < abs(self.CDF[idx]-x):
					mod = 1
				val = self.CDF[idx-mod]

				self.CDF[idx-mod] = (1-rate_w)*val + rate_w*x

				# Attractor - stability problems
				for i in range(len(self.CDF)):
					if i != idx-mod:
						d = i-(idx-mod)
						d /= self.N_bins
						w = gexp(d,.05)
						self.CDF[i] = (1-attractor_w)*self.CDF[i] + attractor_w*(self.CDF[i]*(1-w)+x*w)

				# Smoothing
				CDF_t = self.CDF.copy()
				for i in range(1,len(self.CDF)-1):
					self.CDF[i] = (1-smooth_w)*self.CDF[i] + (smooth_w)*(.5*CDF_t[i-1]+.5*CDF_t[i+1])








# douglas and meng : 1994 : 
#  
#
#
