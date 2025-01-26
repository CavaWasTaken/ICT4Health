#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erfc, erfcinv
#%%
#np.random.seed(30)  # set the seed to reproduce the experiment
plt.close('all')
N=20 # number of independent uniform random variables
Nsamp=1000
xUnif=np.random.rand(Nsamp,N)-0.5 # uniformly distributed in -0.5,0.5 (variance = 1/12)
mu=0
sigma=np.sqrt(1/12)

x=np.sum(xUnif,axis=1)/np.sqrt(N)
plt.figure()
plt.subplot(2,1,1)
u=plt.hist(xUnif[:,0],bins=100,density=True,label='uniform random variable')
pdf=stats.uniform.pdf(x = u[1],loc=-0.5, scale=1)#Compute the theoretical uniform pdf
plt.plot(u[1],pdf,label='theoretical uniform')
plt.legend()
plt.subplot(2,1,2)
a=plt.hist(x,bins=100,density=True,label='sum of '+str(N)+' uniform random variables')
pdf = stats.norm.pdf(x = a[1], loc=mu, scale=sigma) #Compute the theoretical Gaussian pdf
plt.plot(a[1],pdf,label='theoretical Gaussian')
plt.legend()

#%% measured versus theoretical CDF
xs = np.sort(x)
y = np.arange(Nsamp)/Nsamp
plt.figure()
plt.plot(xs,y,label='measured')
plt.plot(xs,1-0.5*erfc((xs-mu)/(np.sqrt(2)*sigma)),label='theory')
plt.legend()
plt.grid()
#%% q-q plot
percentiles = np.arange(0,99,1)
qs=percentiles/100
ii = np.floor(qs*Nsamp).astype(int)
xqs = (xs[ii]+xs[ii+1])/2
xqs_theory = np.sqrt(2)*sigma*erfcinv(2*(1-qs))+mu
plt.figure()
plt.plot(xqs_theory,xqs,'-o',markersize=4)
plt.plot(xqs_theory,xqs_theory,'r',linewidth=2)
plt.xlabel('x_q (theory)')
plt.ylabel('x_q (meas)')
plt.grid()
#%% z-score
z=(np.mean(x)-mu)/sigma*np.sqrt(Nsamp)
p_value = erfc(z/np.sqrt(2))
print('p-value for the z-score is ',p_value)
#%% t-score
def tsco(x):
    m = np.mean(x)
    s = np.sum((x-m)**2)/(Nsamp-1)
    t = (m-mu)/s*np.sqrt(Nsamp)
    return t
t = tsco(x)
print('absolute value of t-score for x is ',np.abs(t))
#%% excess kurtosis
def exc_kurt(x):
    N=len(x)
    m=np.sum(x)/N
    s4=np.sum((x-m)**4)/N
    s2=np.sum((x-m)**2)/N
    k0=s4/s2**2
    A = (N-1)/(N-2)/(N-3)*((N+1)*k0-3*(N-1));
    return A
ku = exc_kurt(x)
print('absolute vlue of excess kurtosis for x is ',np.abs(ku))
#%% Anderson-Darling test
def A_D(x,mu,sigma):
# -N -\sum_{i=1}^{N}\frac{2i-1}{N} \left[ \ln
# F(y_i)+\ln(1-F(y_{N+1-i}))\right]
    N = len(x)
    xsd = -np.sort(-x)
    Fd = 1-0.5*erfc((xsd-mu)/np.sqrt(2)/sigma);
    Fu = np.flipud(Fd);
    ii=np.arange(1,N+1)
    a=(2*ii-1)/N;
    out = -N-a@(np.log(Fu)+np.log(1-Fd));
    return out
a2 = A_D(x,mu,sigma)
print('Anderson-Darling test a^2 for x is ',a2)
#%% estimate p-values for the chosen statistics using simulation
Nexp = 1000#number of experiments to generate the p-value curve
valT = np.zeros((Nexp,))
valKurt = np.zeros((Nexp,))
valA_D = np.zeros((Nexp,))
for k in range(Nexp):
    xg = np.random.randn(Nsamp)*sigma + mu # hypothesis H_0 is satisfied
    valT[k] = tsco(xg)# find t-score for vector xg that satisfies hypothesis H_0
    valKurt[k] = exc_kurt(xg) # find kurtosis for vector xg that satisfies hypothesis H_0
    valA_D[k] = A_D(xg,mu,sigma)# find A_D for vector xg that satisfies hypothesis H_0
#%% final plots to check Gaussianity from p-values (two-tail p-value)
plt.figure()
v1 = np.min(np.abs(valT))
v2 = np.max(np.abs(valT))
plt.semilogy(np.sort(np.abs(valT)),1-np.arange(Nexp)/Nexp,'b',label = 't-score')
plt.semilogy([np.abs(t),np.abs(t)],[1e-5,1],'r--',label = 'measured t-score')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('t-score')

plt.figure()
v1 = np.min(np.abs(valKurt))
v2 = np.max(np.abs(valKurt))
plt.semilogy(np.sort(np.abs(valKurt)),1-np.arange(Nexp)/Nexp,'b',label = 'excess kurtosis')
plt.semilogy([np.abs(ku),np.abs(ku)],[1e-5,1],'r--',label = 'measured excess kurtosis')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('Excess kurtosis')

plt.figure()
v1 = np.min(np.abs(valA_D))
v2 = np.max(np.abs(valA_D))
plt.semilogy(np.sort(np.abs(valA_D)),1-np.arange(Nexp)/Nexp,'b',label = 'Anderson-Darling test')
plt.semilogy([a2,a2],[1e-5,1],'r--',label = 'measured A-D metric')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('Anderson-Darling test')

    
