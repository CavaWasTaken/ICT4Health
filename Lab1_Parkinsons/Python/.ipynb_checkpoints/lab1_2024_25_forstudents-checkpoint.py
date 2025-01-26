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
os.chdir('/home/cavallinux/Backup/Magistrale/ICT4Health/Lab1_Parkinsons/Python')

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
Nte=Np-Ntr   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series)
ss=X_tr.std()# standard deviation (series)
my=mm['total_UPDRS']# mean of total UPDRS
sy=ss['total_UPDRS']# st.dev of total UPDRS
#%% Generate the normalized training and test datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss# normalized data with the mean and st.dev. of the training data .This helps the algorithm to converge faster
ysh_norm=Xsh_norm['total_UPDRS']# regressand only, is the value that we want to predict
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#'],axis=1)# regressors only, are the values that we use to predict the regressand
regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors
print("The new regressors are: ",len(regressors))
#print(regressors)
Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray
X_tr_norm=Xsh_norm[0:Ntr] # regressors for training phase
X_te_norm=Xsh_norm[Ntr:] # regressors for test phase
y_tr_norm=ysh_norm[0:Ntr] # regressand for training phase
y_te_norm=ysh_norm[Ntr:] #regressand for test phase
print(X_tr_norm.shape,X_te_norm.shape)
#%% LLS regression
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)  # we use the features X (regressors) and the target value Y (total_UPDRS - our regressand), in order to find the weights w_hat that compute y = X*w_hat
y_hat_tr_norm=X_tr_norm@w_hat   # we use the weights w_hat to predict new values of Y based on the features X that we have, for the training dataset
y_hat_te_norm=X_te_norm@w_hat   # do the same for the test dataset (@ is the matrix multiplication operator)
#%% de-normalize data
y_tr=y_tr_norm*sy+my    # de-normalize the training regressand, multiply by the st.dev. and add the mean
y_te=y_te_norm*sy+my    # de-normalize the test regressand, multiply by the st.dev. and add the mean
y_hat_tr=y_hat_tr_norm*sy+my    # de-normalize the training prediction, multiply by the st.dev. and add the mean
y_hat_te=y_hat_te_norm*sy+my    # de-normalize the test prediction, multiply by the st.dev. and add the mean
#%% plot the optimum weight vector for LLS
nn=np.arange(Nf)    # number of features
plt.figure(figsize=(6,4))
plt.plot(nn,w_hat,'-o') # plot the weights
ticks=nn    # x-axis ticks are the features
plt.xticks(ticks, regressors, rotation=90)  # set the x-axis ticks
plt.ylabel(r'$\^w(n)$') # set the y-axis label with Latex notation for the weights
plt.title('LLS-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./Plot/LLS-what.png')
plt.draw()
#%% plot the error histograms, difference between the true and predicted values
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
plt.savefig('./Plot/LLS-yhat_vs_y.png')
plt.draw()

#%% statistics of the errors
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
print(results)
# plt.show()

#%% my code
matricola = [3, 4, 6, 7, 4, 2]  # values of my matricola, i need to select one of them randomly
randomSeed = random.choice(matricola)   # select one of the values randomly

Xsh_norm_=(Xsh-mm)/ss# normalized data with the mean and st.dev. of the training data .This helps the algorithm to converge faster
ysh_norm_=Xsh_norm_['total_UPDRS']# regressand only, is the value that we want to predict
user_input = int(input("Insert 0 if you want to consider Motor UPDRS as parameter, 1 if you don't: "))
if user_input == 0:
    print("Motor UPDRS is considered as parameter")
    Xsh_norm_=Xsh_norm_.drop(['Jitter:DDP', 'Shimmer:DDA', 'total_UPDRS','subject#'],axis=1)# drop unwanted regressors
else:
    print("Motor UPDRS is not considered as parameter")
    Xsh_norm_=Xsh_norm_.drop(['Jitter:DDP', 'Shimmer:DDA', 'total_UPDRS','subject#', 'motor_UPDRS'],axis=1)# drop unwanted regressors
regressors=list(Xsh_norm_.columns)
Nf = len(regressors) # number of regressors
print("Excercise:\nThe new regressors are: ",len(regressors))

Xsh_norm_=Xsh_norm_.values # from dataframe to Ndarray
ysh_norm_=ysh_norm_.values # from dataframe to Ndarray
X_tr_norm_=Xsh_norm_[0:Ntr] # regressors for training phase
X_te_norm_=Xsh_norm_[Ntr:] # regressors for test phase
y_tr_norm_=ysh_norm_[0:Ntr] # regressand for training phase
y_te_norm_=ysh_norm_[Ntr:] #regressand for test phase

#%%Steepes descent
w_hat_=np.random.randn(Nf) # random initialization of the weights
n_iter=1000 # number of iterations
min_loss = 1e-5 # minimum loss
prev_loss = 0 # previous loss
alpha=0.001 # learning rate
for i in range(n_iter):
    y_hat_tr_norm_=X_tr_norm_@w_hat_   # prediction
    E_tr_=y_tr_norm_-y_hat_tr_norm_    # error
    loss = np.mean(E_tr_**2)    # loss
    if np.abs(prev_loss - loss) < min_loss:    # check if the loss change is below the threshold
        print(f"Early stopping at iteration {i}.")
        break
    prev_loss = loss    # update the previous loss
    gradient = (1/Nc) * (X_tr_norm_.T @ E_tr_)    # gradient
    w_hat_+=alpha*gradient  # weight update

#%% de-normalize data
y_tr_=y_tr_norm_*sy+my    # de-normalize the training regressand, multiply by the st.dev. and add the mean
y_te_=y_te_norm_*sy+my    # de-normalize the test regressand, multiply by the st.dev. and add the mean
y_hat_tr_=y_hat_tr_norm_*sy+my    # de-normalize the training prediction, multiply by the st.dev. and add the mean
y_hat_te_=X_te_norm_@w_hat_*sy+my    # de-normalize the test prediction, multiply by the st.dev. and add the mean

#%% plot the optimum weight vector for Steepest descent
nn=np.arange(Nf)    # number of features
plt.figure(figsize=(6,4))
plt.plot(nn,w_hat_,'-o') # plot the weights
ticks=nn    # x-axis ticks are the features
plt.xticks(ticks, regressors, rotation=90)  # set the x-axis ticks
plt.ylabel(r'$\^w(n)$') # set the y-axis label with Latex notation for the weights
plt.title('Steepest descent-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./Plot/SteepesDescent-what.png')

#%% plot the error histograms, difference between the true and predicted values
E_tr_=(y_tr_-y_hat_tr_)# training
E_te_=(y_te_-y_hat_te_)# test
M=np.max([np.max(E_tr_),np.max(E_te_)])   # maximum value of the errors on the training and test dataset
m=np.min([np.min(E_tr_),np.min(E_te_)])   # minimum value of the errors on the training and test dataset
common_bins=np.arange(m,M,(M-m)/50) # prepare the bins for the histogram
e=[E_tr_,E_te_]   # list of errors
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['training','test']) # plot the histograms
plt.xlabel(r'$e=y-\^y$')    # set the x-axis label with Latex notation for the error
plt.ylabel(r'$P(e$ in bin$)$')  # set the y-axis label with Latex notation for the probability of the error in the bin
plt.legend()
plt.grid()
plt.title('Steepest descent-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./Plot/SteepesDescent-hist.png')

#%% plot the regression lines
plt.figure(figsize=(4,4))
plt.plot(y_te_,y_hat_te_,'.',label='all') # plot the test dataset
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)   # plot the bisector line (y=x)
plt.xlabel(r'$y$')  # set the x-axis label with Latex notation for the true value of the regressand
plt.axis('square')  # set the aspect ratio of the plot to be square
plt.ylabel(r'$\^y$')    # set the y-axis label with Latex notation for the predicted value of the regressand
plt.grid()
plt.title('Steepest descent-test')
plt.tight_layout()
plt.savefig('./Plot/SteepesDescent-yhat_vs_y.png')  # this graph shows the predicted values vs the true values
plt.draw()

#%% statistics of the errors
E_tr_max=E_tr_.max() # maximum error on the training dataset
E_tr_min=E_tr_.min() # minimum error on the training dataset
E_tr_mu=E_tr_.mean() # mean error on the training dataset
E_tr_sig=E_tr_.std() # standard deviation of the error on the training dataset
E_tr_MSE=np.mean(E_tr_**2)   # mean square error on the training dataset
R2_tr=1-E_tr_MSE/(np.var(y_tr_)) # R^2 coefficient on the training dataset, indicates how well the regression line approximates the real data
c_tr=np.mean((y_tr_-y_tr_.mean())*(y_hat_tr_-y_hat_tr_.mean()))/(y_tr_.std()*y_hat_tr_.std()) # correlation coefficient on the training dataset
E_te_max=E_te_.max() # do the same for the test dataset
E_te_min=E_te_.min()
E_te_mu=E_te_.mean()
E_te_sig=E_te_.std()
E_te_MSE=np.mean(E_te_**2)
R2_te=1-E_te_MSE/(np.var(y_te_))
c_te=np.mean((y_te_-y_te_.mean())*(y_hat_te_-y_hat_te_.mean()))/(y_te_.std()*y_hat_te_.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']    # columns of the dataframe of errors
rows=['Training','test']    # rows of the dataframe with the different datasets
p=np.array([
    [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
    [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
            ])  # values to plot
results=pd.DataFrame(p,columns=cols,index=rows) # create the dataframe
print(results)   # print the dataframe
# plt.show() # show the plotss

#only using the voice parameters for the prediction of the total UPDRS is not a good idea, because the correlation coefficient is very low. We can see this on the plot that are very noisy