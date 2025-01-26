
# FastICA on an artificial example

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from matplotlib.pyplot import cm

np.random.seed(50)  # set the seed to reproduce the experiment

plt.close('all')
#%% Generate the signals
N = 10000 # number opf samples for each signal
N_signals=4 # number of signals
t = np.linspace(0, 10, N) # generate the time axis (seconds)
t.resize(1,N)   # t has shape (N,) we need (1,N)
freq1 = 0.5 # main frequency (Hz)
freq2 = freq1*np.sqrt(2)
freq3 = freq1*np.sqrt(5)
x1 = np.sin(2*np.pi*freq1* t-np.pi/4)  # Signal 1: sinusoidal signal of frequency freq1 and phase shift of pi/4
x2 = np.sign(np.sin(2*np.pi*freq2* t-np.pi/5))  # Signal 2: square wave signal of frequency freq2 and phase shift of pi/5
x3 = signal.sawtooth(2*np.pi*freq3* t)  # Signal 3: saw tooth signal of frequency freq3 and without phase shift
x4=np.cumsum(x2)    # Signal 4: integral of signal 2
x4=(x4-x4.mean())/x4.std()  # standardize the signal
x4=x4/x4.max()  # normalize the signal
x4.resize(1,N) # signal 2: triangular signal
t=t.flatten()   # t has shape (1,N) we need (N,)
# our four signals are far from Gaussian pdf
# if you want to see what happens with signals having Gaussian pdf
# x4=np.random.randn(1,N)
# x4=x4/x4.max()
# x3=np.random.randn(1,N)
# x3=x3/x3.max()

X = np.concatenate((x1,x2,x3,x4),axis=0) # shape: 4 rows N columns
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()# plots of the original signals
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'sinusoidal signal'
    elif n == 1:
        label = 'square wave signal'
    elif n == 2:
        label = 'saw tooth signal'
    else:
        label = 'triangular signal'
    plt.plot(t,X[n,:],'--',color=c,label=f'{label}')
    plt.grid()
    plt.legend()
plt.xlabel('t (s)')
plt.tight_layout()
plt.show()

#%% Generate the observed/mixed signals
A = np.random.randn(N_signals,N_signals)  # true weight matrix (random) - noise 
Y = np.dot(A,X)  # observed/mixed signals, shape: 4 rows N columns  - apply the noise to the original signals
W = np.linalg.inv(A)    # true mixing matrix
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))  # colors for the plots
plt.figure()# plots of the mixed signals
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'mixed sinusoidal signal'
    elif n == 1:
        label = 'mixed square wave signal'
    elif n == 2:
        label = 'mixed saw tooth signal'
    else:
        label = 'mixed triangular signal'
    plt.plot(t,Y[n,:],'--',color=c,label=f'{label}')
    plt.grid()
    plt.legend()
plt.xlabel('t (s)')
plt.tight_layout()
plt.show()
#%% reshape
X=X.T #shape: N rows, 4 columns 
Y=Y.T #shape: N rows, 4 columns 
#%% look at the pdfs
plt.figure(figsize=(5,8))# histograms
Nbins = np.ceil(1+np.log2(N)).astype(int)   # number of bins for the histograms
plt.title('normalized histograms of original signals')
for n in range(N_signals):
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'sinusoidal signal'
    elif n == 1:
        label = 'square wave signal'
    elif n == 2:
        label = 'saw tooth signal'
    else:
        label = 'triangular signal'
    plt.hist(X[:,n],bins=Nbins,density=True,label=f'Comp. {label}')
    plt.legend()
plt.xlabel('sample values')
plt.ylabel('estimated pdf')
plt.tight_layout()
plt.figure(figsize=(5,8))
plt.title('normalized histograms of mixed signals')
for n in range(N_signals):
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'sinusoidal signal'
    elif n == 1:
        label = 'square wave signal'
    elif n == 2:
        label = 'saw tooth signal'
    else:
        label = 'triangular signal'
    plt.hist(Y[:,n],bins=Nbins,density=True,label=f'Comp. {label}')
    plt.legend()
plt.xlabel('sample values')
plt.ylabel('estimated pdf')
plt.tight_layout()
plt.show()
# note that the histograms of the original signals are far from Gaussian pdf
# while the histograms of the mixed signals are more similar to Gaussian pdf
#%% Use FastICA
ica = FastICA(n_components=N_signals,algorithm="deflation", whiten="unit-variance")
XhatICA = ica.fit_transform(Y)  # Reconstruct indep signals from obervations
Ahat = ica.mixing_  # estimated A
What = ica.components_ # estimated matrix W: Xhat=np.dot(Y,ica.components_.T)

vm1=W.min() # min value of the true matrix W
VM1=W.max() # max value of the true matrix W
vm2=What.min()  # min value of the estimated matrix W
VM2=What.max()  # max value of the estimated matrix W
vm=np.min([vm1,vm2])    # min value for the colormap
VM=np.max([VM1,VM2])    # max value for the colormap

plt.figure()
plt.subplot(1,2,1)
plt.matshow(W,0,vmin=vm,vmax=VM, cmap='hot')    # plot the true matrix W
plt.colorbar()
plt.title('True matrix W')
plt.subplot(1,2,2)
plt.matshow(What,0,vmin=vm,vmax=VM, cmap='hot') # plot the estimated matrix W
plt.colorbar()
plt.title('Estimated matrix W')
plt.show()
# heat plot show the matrix and its cells, and each cell is colored according to its value, 
# a darker color indicates a lower value and a lighter color indicates a higher value
# the color bar helps us to understand the values of the matrix
#%% Use also PCA
pca = PCA(n_components=N_signals)   # reduce the dimensionality to N_signals
XhatPCA = pca.fit_transform(Y)  # Reconstruct signals based on orthogonal components
#%% Plot the results
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'sinusoidal signal'
    elif n == 1:
        label = 'square wave signal'
    elif n == 2:
        label = 'saw tooth signal'
    else:
        label = 'triangular signal'
    plt.plot(t,XhatICA[:,n]/np.max(XhatICA[:,n]),'-',color=c,label=f'ICA component {label}') 
    plt.plot(t,X[:,n]/np.max(X[:,n]),'--',color=c,label=f'{label}')
    plt.grid()
    plt.legend()
plt.xlabel('t (s)')
plt.show()
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    if n == 0:
        label = 'sinusoidal signal'
    elif n == 1:
        label = 'square wave signal'
    elif n == 2:
        label = 'saw tooth signal'
    else:
        label = 'triangular signal'
    plt.plot(t,XhatPCA[:,n]/np.max(XhatPCA[:,n]),'-',color=c,label=f'PCA component {label}') 
    plt.plot(t,X[:,n]/np.max(X[:,n]),'--',color=c,label=f'{label}')
    plt.grid()
    plt.legend()
plt.xlabel('t (s)')
plt.show()