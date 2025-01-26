# -*- coding: utf-8 -*-
"""

@author: Monica Visintin

Regress Total UPDRS from the other features in file "parkinsons_updrs.csv"

"""
#%% import libraries
import os # import the os library, useful to set the working directory
import pandas as pd # import the Pandas library, useful for data manipulation
import numpy as np  # import the Numpy library, useful for numerical operations
import matplotlib.pyplot as plt # import the Matplotlib library, useful for plotting
import random # import the random library, useful for random number generation

# set the working directory
os.chdir('/home/cavallinux/Backup/Magistrale/ICT4Health/Lab2_Parkinsons/Python')

pd.set_option('display.precision', 3)  # set the precision for the output of the Pandas dataframe
#%% Read the input csv file
plt.close('all') # close all the figures that might still be open from previous runs
X=pd.read_csv("/home/cavallinux/Backup/Magistrale/ICT4Health/Lab1_Parkinsons/Python/parkinsons_updrs_av.csv") # read the dataset; x is a Pandas dataframe
features=list(X.columns)    # list of features in the dataset
subj=pd.unique(X['subject#'])   # existing values of patient ID
print("The original dataset shape  is ",X.shape)
print("The number of distinct patients in the dataset is ",len(subj))
print("the original dataset features are ",len(features))
print(features)

Np,Nc=X.shape# Np = number of rows/measurements Nc=number Nf of regressors + 1 (regressand total UPDRS is included)
#%% Have a look at the dataset
print(X.describe().T) # gives the statistical description of the content of each column
print(X.info()) # gives you information about the data (number of valid values , type)
#%% Measure and show the covariance matrix
Xnorm=(X-X.mean())/X.std()# normalized data
c=Xnorm.cov()# note: xx.cov() gives the wrong result
# the covariance matrix rappresents the correlation between each couple of features (> 0 the features are positively correlated, < 0 they are negatively correlated, = 0 they are not correlated)
plt.figure()
plt.matshow(np.abs(c.values),fignum=0)# absolute value of corr.coeffs
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)
plt.colorbar()
plt.title('Correlation coefficients of the features')
plt.tight_layout()
plt.savefig('./Plot/corr_coeff.png') # save the figure
plt.draw()
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)),features,rotation=90)
plt.title('Corr. coeff. between total_UPDRS and the other features')
plt.tight_layout()
plt.draw()
plt.savefig('./Plot/UPDRS_corr_coeff.png') # save the figure
#%% Shuffle the data (two out of many methods). This is important in order to avoid that the training set is composed only by the first patients
# first method:
np.random.seed(30) # set the seed for random shuffling
# indexsh=np.arange(Np) # generate array [0,1,...,Np-1]
# np.random.shuffle(indexsh) # shuffle the array
# Xsh=X.copy()
# Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False) # shuffle accordingly the dataframe
# Xsh=Xsh.sort_index(axis=0) # reset index of the dataframe
# comment: Xsh.reset_index() exists, but a further index column would be created
# second method
Xsh=X.sample(frac=1, replace=False, random_state=30, axis=0, ignore_index=True)
#Xsh=X.sample(frac=1, replace=False, axis=0, ignore_index=True)

#%% Generate training and test matrices
Ntr=int(Np*0.5)  # number of training points
Nval=int(Np*0.25) # number of validation points
Nte=(Np*0.25)   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series)
ss=X_tr.std()# standard deviation (series)
my=mm['total_UPDRS']# mean of total UPDRS
sy=ss['total_UPDRS']# st.dev of total UPDRS
#%% Generate the normalized training and test datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss# normalized data with the mean and st.dev. of the training data .This helps the algorithm to converge faster
ysh_norm=Xsh_norm['total_UPDRS']# regressand only, is the value that we want to predict
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#', 'age'],axis=1)# regressors only, are the values that we use to predict the regressand
regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors
print("The new regressors are: ",len(regressors))
#print(regressors)
Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray
X_tr_norm=Xsh_norm[0:Ntr] # regressors for training phase
X_val_norm=Xsh_norm[Ntr:Ntr+Nval] # regressors for validation phase
X_te_norm=Xsh_norm[Ntr+Nval:] # regressors for test phase
y_tr_norm=ysh_norm[0:Ntr] # regressand for training phase
y_val_norm=ysh_norm[Ntr:Ntr+Nval] #regressand for validation phase
y_te_norm=ysh_norm[Ntr+Nval:] #regressand for test phase
print(X_tr_norm.shape,X_val_norm.shape,X_te_norm.shape)
#%%LLS considering K nearest neighbors
# K=1 # number of nearest neighbors
# eps=10**-8
# E_val_MSE_list = []
# for x in X_val_norm: # for each sample in the validation dataset
#     dist=np.sum((X_tr_norm-x)**2,axis=1)    # compute the Euclidean distance between the sample x and all the samples in the training dataset
#     ind=np.argsort(dist)    # sort the distances in ascending order and return the indices of the sorted array
#     w_hat=np.linalg.inv(X_tr_norm[ind[0:K]].T@X_tr_norm[ind[0:K]]+eps*np.eye(Nf))@(X_tr_norm[ind[0:K]].T@y_tr_norm[ind[0:K]])  # we use the features X (regressors) and the target value Y (total_UPDRS - our regressand), in order to find the weights w_hat that compute y = X*w_hat
#     y_hat_val_norm=X_val_norm@w_hat   # y = X * w_hat
#     E_val=(y_val_norm-y_hat_val_norm)    # compute the error between the true and predicted values
#     E_val_MSE=np.mean(E_val**2) # compute the mean square error
#     E_val_MSE_list.append(E_val_MSE)    # append the error to the list
# E_val_MSE_mean=np.mean(E_val_MSE_list) # compute the mean of the errors
# plt.figure()
# plt.plot(E_val_MSE_list, marker='o')
# plt.xlabel('Sample Index')
# plt.ylabel('E_val_MSE')
# plt.title('Validation Mean Square Error (E_val_MSE)')
# plt.grid(True)
# plt.savefig('./Plot/E_val_MSE_minimum_K.png') # save the figure

Kmin = 10    # minimum number of nearest neighbors
Kmax = Nf   # maximum number of nearest neighbors
mean_MSE_per_K = [] # list to store the mean MSE for each K
eps = 10 ** -8  # small value to avoid singular matrix
for K in range(Kmin, Kmax + 1): # for each K value  from Kmin to Kmax
    E_val_MSE_list = [] # list to store the MSE for each sample in the validation dataset
    for x in X_val_norm:    # for each sample in the validation dataset
        # x is a vector of features for a single sample in the validation dataset
        dist = np.sum((X_tr_norm - x) ** 2, axis=1) # compute the Euclidean distance between the sample x and all the samples in the training dataset
        ind = np.argsort(dist)  # sort the distances in ascending order and return the indices of the sorted array
        w_hat = np.linalg.inv(X_tr_norm[ind[0:K]].T @ X_tr_norm[ind[0:K]] + eps * np.eye(Nf)) @ (X_tr_norm[ind[0:K]].T @ y_tr_norm[ind[0:K]])   # we use the features X (regressors) and the target value Y (total_UPDRS - our regressand), in order to find the weights w_hat that compute y = X*w_hat
        y_hat_val_norm = X_val_norm @ w_hat # y = X * w_hat
        y_hat_val = y_hat_val_norm * sy + my # denormalize the predicted value
        y_val = y_val_norm * sy + my # denormalize the true value
        E_val = y_val - y_hat_val   # compute the error between the true and predicted values
        E_val_MSE = np.mean(E_val ** 2) # compute the mean square error
        E_val_MSE_list.append(E_val_MSE)    # append the error to the list
    mean_MSE_per_K.append(np.mean(E_val_MSE_list))  # append the mean MSE for the current K value to the list

plt.figure()
plt.plot(range(Kmin, Kmax + 1), mean_MSE_per_K, marker='o')
plt.xlabel('K value')
plt.ylabel('Mean E_val_MSE')
plt.title('Mean Validation Mean Square Error (E_val_MSE) for each K')
plt.grid(True)
plt.savefig('./Plot/mean_E_val_MSE_per_K.png')

K_opt = np.argmin(mean_MSE_per_K) + Kmin   # optimal K value, +1 because the index starts from 0
print("The optimal K value is: ", K_opt)

#%% Evaluate the performance of the model on the test dataset
E_te_MSE_list = []
Errors = []
for x in X_te_norm: # for each sample in the test dataset
    dist=np.sum((X_tr_norm-x)**2,axis=1)    # compute the Euclidean distance between the sample x and all the samples in the training dataset
    ind=np.argsort(dist)    # sort the distances in ascending order and return the indices of the sorted array
    w_hat=np.linalg.inv(X_tr_norm[ind[0:K_opt]].T@X_tr_norm[ind[0:K_opt]]+eps*np.eye(Nf))@(X_tr_norm[ind[0:K_opt]].T@y_tr_norm[ind[0:K_opt]])  # we use the features X (regressors) and the target value Y (total_UPDRS - our regressand), in order to find the weights w_hat that compute y = X*w_hat
    y_hat_te_norm=X_te_norm@w_hat   # y = X * w_hat
    y_hat_te = y_hat_te_norm*sy+my # denormalize the predicted value
    y_te = y_te_norm*sy+my # denormalize the true value
    E_te=(y_te-y_hat_te)    # compute the error between the true and predicted values
    Errors.append(E_te) # append the error to the list
    E_te_MSE = np.mean(E_te ** 2) # compute the mean square error
    E_te_MSE_list.append(E_te_MSE)    # append the error to the list
    
# at each iteration i have selected a sample x from the test dataset, found the eucliendean distance between x and all the other samples in the training dataset, 
# sorted the distances in ascending order and selected the K nearest neighbors, computed the weights w_hat that compute the predicted value y_hat_te, denormalized the predicted 
# and true values, computed the error between the true and predicted values and appended the error to the list, computed the mean square error and appended the error to the list
# So Errors is an array of arrays, where each array contains the errors for each sample in the test dataset. So in order of calculate the mean error, standard deviation and square error
# I have to flatten the list of errors and then compute the mean, standard deviation and square error
plt.figure()
plt.plot(E_te_MSE_list, marker='o')
plt.xlabel('Sample Index')
plt.ylabel('E_te_MSE')
plt.title('Test Mean Square Error (E_te_MSE)')
plt.grid(True)
plt.savefig('./Plot/E_te_MSE_optimal_K.png') # save the figure

#%% mean error, standard deviation and square error
Errors = np.array(Errors)   # convert the list of errors to a numpy array
mean_error = np.mean(Errors)    # mean error
std_error = np.std(Errors)      # standard deviation of the error   
square_error = np.mean(Errors ** 2)   # mean square error
min_error = np.min(Errors)  # minimum error
max_error = np.max(Errors)  # maximum error
R2 = 1 - square_error / np.var(y_te)    # R^2 coefficient, indicates how well the regression line approximates the real data
corr = np.mean((y_te - y_te.mean()) * (y_hat_te - y_hat_te.mean())) / (y_te.std() * y_hat_te.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']    # columns of the dataframe of errors
rows=['test']    # rows of the dataframe with the different datasets
p=np.array([
    [min_error,max_error,mean_error,std_error,square_error,R2,corr]
])    # values of the dataframe
results=pd.DataFrame(p,columns=cols,index=rows) # create the dataframe
print("LLS_K:\n",results)   # print the dataframe

#%% plot the error histograms, difference between the true and predicted values
common_bins=np.arange(min_error,max_error,(max_error-min_error)/50) # prepare the bins for the histogram
e=np.array(Errors).flatten() # flatten the list of errors
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['test']) # plot the histograms
plt.xlabel(r'$e=y-\^y$')    # set the x-axis label with Latex notation for the error
plt.ylabel(r'$P(e$ in bin$)$')  # set the y-axis label with Latex notation for the probability of the error in the bin
plt.legend()
plt.grid()
plt.title('LLS-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./Plot/LLS_K-hist.png')
plt.draw()

#%% plot the regression lines
plt.figure(figsize=(4,4))
plt.plot(y_te,y_hat_te,'.',label='all') # plot the test dataset
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)   # plot the bisector line (y=x)
plt.xlabel(r'$y$')  # set the x-axis label with Latex notation for the true value of the regressand
plt.axis('square')  # set the aspect ratio of the plot to be square
plt.ylabel(r'$\^y$')    # set the y-axis label with Latex notation for the predicted value of the regressand
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./Plot/LLS_K-yhat_vs_y.png')
plt.draw()

#%% second part - LLS with 75% of training dataset
Ntr = int(Np * 0.75)  # number of training points
Nte = 1 - Ntr  # number of test points

X_tr = Xsh[0:Ntr]  # dataframe that contains only the training data
mm = X_tr.mean()  # mean (series)
ss = X_tr.std()  # standard deviation (series)
my = mm['total_UPDRS']  # mean of total UPDRS
sy = ss['total_UPDRS']  # st.dev of total UPDRS

Xsh_norm=(Xsh-mm)/ss# normalized data with the mean and st.dev. of the training data .This helps the algorithm to converge faster
ysh_norm=Xsh_norm['total_UPDRS']# regressand only, is the value that we want to predict
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#', 'age'],axis=1)# regressors only, are the values that we use to predict the regressand
regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors

Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray
X_tr_norm=Xsh_norm[0:Ntr] # regressors for training phase
X_te_norm=Xsh_norm[Ntr:] # regressors for test phase
y_tr_norm=ysh_norm[0:Ntr] # regressand for training phase
y_te_norm=ysh_norm[Ntr:] #regressand for test phase
print(X_tr_norm.shape,X_te_norm.shape)

w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)  # we use the features X (regressors) and the target value Y (total_UPDRS - our regressand), in order to find the weights w_hat that compute y = X*w_hat
y_hat_tr_norm=X_tr_norm@w_hat   # we use the weights w_hat to predict new values of Y based on the features X that we have, for the training dataset
y_hat_te_norm=X_te_norm@w_hat   # do the same for the test dataset (@ is the matrix multiplication operator)

y_tr=y_tr_norm*sy+my    # de-normalize the training regressand, multiply by the st.dev. and add the mean
y_te=y_te_norm*sy+my    # de-normalize the test regressand, multiply by the st.dev. and add the mean
y_hat_tr=y_hat_tr_norm*sy+my    # de-normalize the training prediction, multiply by the st.dev. and add the mean
y_hat_te=y_hat_te_norm*sy+my    # de-normalize the test prediction, multiply by the st.dev. and add the mean

E_tr=(y_tr-y_hat_tr)# training
E_te=(y_te-y_hat_te)# test
M=np.max([np.max(E_tr),np.max(E_te)])   # maximum value of the errors on the training and test dataset
m=np.min([np.min(E_tr),np.min(E_te)])   # minimum value of the errors on the training and test dataset
common_bins=np.arange(m,M,(M-m)/50) # prepare the bins for the histogram
e=[E_tr,E_te]   # list of errors
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['training','test']) # plot the histograms
plt.xlabel(r'$e=y-\^y$')    # set the x-axis label with Latex notation for the error
plt.ylabel(r'$P(e$ in bin$)$')  # set the y-axis label with Latex notation for the probability of the error in the bin
plt.legend()
plt.grid()
plt.title('LLS-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./Plot/LLS-hist.png')
plt.draw()

plt.figure(figsize=(4,4))
plt.plot(y_te,y_hat_te,'.',label='all') # plot the test dataset
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)   # plot the bisector line (y=x)
plt.xlabel(r'$y$')  # set the x-axis label with Latex notation for the true value of the regressand
plt.axis('square')  # set the aspect ratio of the plot to be square
plt.ylabel(r'$\^y$')    # set the y-axis label with Latex notation for the predicted value of the regressand
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./Plot/LLS-yhat_vs_y.png')
plt.draw()

E_tr_max=E_tr.max() # maximum error on the training dataset
E_tr_min=E_tr.min() # minimum error on the training dataset
E_tr_mu=E_tr.mean() # mean error on the training dataset
E_tr_sig=E_tr.std() # standard deviation of the error on the training dataset
E_tr_MSE=np.mean(E_tr**2)   # mean square error on the training dataset
R2_tr=1-E_tr_MSE/(np.var(y_tr)) # R^2 coefficient on the training dataset, indicates how well the regression line approximates the real data
c_tr=np.mean((y_tr-y_tr.mean())*(y_hat_tr-y_hat_tr.mean()))/(y_tr.std()*y_hat_tr.std()) # correlation coefficient on the training dataset
E_te_max=E_te.max() # do the same for the test dataset
E_te_min=E_te.min()
E_te_mu=E_te.mean()
E_te_sig=E_te.std()
E_te_MSE=np.mean(E_te**2)
R2_te=1-E_te_MSE/(np.var(y_te))
c_te=np.mean((y_te-y_te.mean())*(y_hat_te-y_hat_te.mean()))/(y_te.std()*y_hat_te.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']    # columns of the dataframe of errors
rows=['Training','test']    # rows of the dataframe with the different datasets
p=np.array([
    [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
    [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
            ])  # values to plot

results=pd.DataFrame(p,columns=cols,index=rows)
print("LLS:\n", results)   # print the dataframe