import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import DBSCAN

#%%
plt.close('all')
xx=pd.read_csv("covid_serological_results.csv")
xx=xx[xx.COVID_swab_res!=1]# remove unclear results
xx.COVID_swab_res[xx.COVID_swab_res==2]=1# set swab result from 2 to 1 for ill patients 
#%%
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1=illness
Test1=xx.IgG_Test1_titre.values # level of immoglobulin G for test 1 (values range from 2.5 to 314)
Test2=xx.IgG_Test2_titre.values # level of immoglobulin G for test 2 (values range from 0 to 9.71

print(f"Original dataset:\n{xx.describe()}")
pd.plotting.scatter_matrix(xx, figsize=(10,10))
plt.savefig('./plot/scatter_matrix.png')

#%%Remove outliers
# normalize data with minmax normalization
norm_xx = (xx - xx.min()) / (xx.max() - xx.min())
print(f"Normalized dataset:\n{norm_xx.describe()}")
clusters = DBSCAN(eps=0.1, min_samples=5).fit(norm_xx)
labels = clusters.labels_
xx = xx[labels != -1]   # remove outliers
#%% ROC curve
# x=Test2 # select the test to analyze
# y=swab  # swab is the ground truth
# x0=x[swab==0] # test results for healthy patients
# x1=x[swab==1] # test results for ill patients
Np=np.sum(swab==1) # number of ill patients
Nn=np.sum(swab==0) # number of healthy patients
thresh_test1 = np.linspace(2.5, 314, 500) # example of threshold
thresh_test2 = np.linspace(0, 9.71, 500) # example of threshold

def ROC_curve(thresh, i, x0, x1):
    spec = []   # specificity - P(T_n|H) - True Negative Rate
    sens = []   # sensitivity - P(T_p|D) - True Positive Rate
    fpr = []    # P(T_p|H) - False Positive Rate
    for t in thresh:    
        n1=np.sum(x1>t) # number of true positives for the given thresh
        sens.append(n1/Np) # sensitivity
        n0=np.sum(x0<t) # number of true negatives
        spec.append(n0/Nn) # specificity
        fpr.append(1-spec[-1]) # false positive rate
        # print(f'specificity P(T_n|H) for threshold {t}, Test2 =',spec)
        # print(f'sensitivity P(T_p|D) for threshold {t}, Test2 =',sens)

    # Find the point of intersection between the ROC curve and the balance line
    intersection_idx = np.argmin(np.abs(np.array(sens) - np.array(spec)))
    # plot ROC curve
    plt.figure()
    plt.plot(fpr, sens, label="ROC Curve")
    plt.plot([1,0],[0,1], label="Balance Threshold")
    # the threshold must be decided case by case. In this example we are choosing the threshold that balance the sensitivity and specificity, but there could be cases where 
    # we are more interested in decreasing the fpr or the fnr. Increasing one means that we are increasing the other.
    # so if we move the selected treshold to the right we are increasing the fpr and decreasing the fnr, and vice versa.
    # in this specific case of Covid19, i think that we should be more interested in decreasing the fnr, so we should move the threshold to the left. Because, false positive are not as bad as false negatives.
    # as false negative, cause this person could spread the virus without knowing that he is infected.
    plt.scatter(fpr[intersection_idx], sens[intersection_idx], color='red', label='Balance Point')
    plt.plot([0,1],[0,1], label="Random Guess", linestyle='--')
    # Find the point of intersection between the ROC curve and the balance line
    plt.xlabel('P(T_p|H)')
    plt.ylabel('P(T_p|D)')
    plt.title(f'ROC curve for Test2')
    plt.legend()
    plt.savefig(f'./plot/My_ROC_Test{i}.png')

    # plot sensitivity and specificity against threshold
    plt.figure()
    plt.plot(thresh, sens, label="P(T_p|D)")
    plt.plot(thresh, spec, label="P(T_n|H)")
    plt.scatter(thresh[intersection_idx], sens[intersection_idx], color='red', label='Balance Point')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(f'Sensitivity and Specificity for Test{i}')
    plt.legend()
    plt.savefig(f'./plot/Sensitivity_Specificity_Test{i}.png')

ROC_curve(thresh_test1, 1, Test1[swab==0], Test1[swab==1])
ROC_curve(thresh_test2, 2, Test2[swab==0], Test2[swab==1])

fpr, tpr, thresh = metrics.roc_curve(swab, Test1, pos_label=1)
roc_auc = roc_auc_score(swab, Test1)    # Area under the ROC curve
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([1,0],[0,1], label="Balance Threshold")
plt.plot([0,1],[0,1], label="Random Guess", linestyle='--')
plt.xlabel('P(T_p|H)')
plt.ylabel('P(T_p|D)')
plt.title(f'ROC curve for Test1\nAUC = {roc_auc}')
plt.legend()
plt.savefig(f'./plot/ROC_Test1.png')

fpr, tpr, thresh = metrics.roc_curve(swab, Test2, pos_label=1)
roc_auc = roc_auc_score(swab, Test2)    # Area under the ROC curve
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([1,0],[0,1], label="Balance Threshold")
plt.plot([0,1],[0,1], label="Random Guess", linestyle='--')
plt.xlabel('P(T_p|H)')
plt.ylabel('P(T_p|D)')
plt.title(f'ROC curve for Test2\nAUC = {roc_auc}')
plt.legend()
plt.savefig(f'./plot/ROC_Test2.png')

# test 1 has a better area under the curve than test 2, so it is a better test to predict the illness