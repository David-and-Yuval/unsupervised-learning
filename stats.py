import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import openpyxl
from seaborn import scatterplot as scatter
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support, f1_score
from scipy.spatial import distance_matrix as dist
from fcmeans import FCM
import random
import warnings
from frauds_wrt_time import *
random.seed(9)

credit = pd.read_csv("creditcard.csv")
frauds = credit[credit.Class == 1]
nofrauds = credit[credit.Class == 0]
gt = np.array(credit["Class"])
vecs = np.array(credit[["V"+str(num) for num in range(1,29)]])
vecs_a = np.array(credit[["V"+str(num) for num in range(1,29)]+["Amount"]])
vecs_2 = np.array(credit[["V14","V17"]])
vecs_z = stats.zscore(vecs_a,axis = 0)
v1 = np.array(credit).T
v2 = np.array(frauds).T
v3 = np.array(nofrauds).T

# f, axes = plt.subplots(1, 2, figsize=(11,5))
# sns.distplot(credit["Amount"],ax=axes[0],hist = 100)
# frauds = pd.read_csv("creditcard.csv", index_col = "Class")
# frauds.drop([0],inplace = True)
# file = open("creditcard.csv",newline = "")
lows = []
for i in range(len(vecs)):
    if vecs_a[i][-1] < 100: # or vecs[i][-1] >= 100:
        lows.append(i)
lows = np.array(lows)
vecs_h = np.delete(vecs,lows,0)
gt_h = np.delete(gt,lows)
# sns.distplot(frauds["Amount"],ax = axes[1],hist = 100)
# plt.show()

def frauds_wrt_amount_main():
    sns.set()
    credit_df = pd.read_csv("creditcard.csv")

    all_amounts = credit_df[['Amount']].round(0).astype(int)
    all_amounts['Freq'] = all_amounts.groupby('Amount')['Amount'].transform('count')

    # all_amounts2 = all_amounts.drop(all_amounts[all_amounts.Freq<5].index)
    # print('all_amounts.shape =',all_amounts2.shape)

    positive_amounts = credit_df[(credit_df['Class']==1)][['Amount','Class']].drop(columns=['Class']).round(0).astype(int)
    positive_amounts['Freq'] = positive_amounts.groupby('Amount')['Amount'].transform('count')

    positive_amounts2 = positive_amounts.drop(positive_amounts[positive_amounts.Freq<2].index)

    print("positive_amounts2.shape =",positive_amounts2.shape)

    plt.figure()
    # plt.subplot(3,1,1)
    # chart=sns.barplot(x='Amount',y='Freq',data=positive_amounts)
    # chart.set_xticklabels(chart.get_xticklabels(),rotation=45,fontweight='light',fontsize=6)
    #
    # plt.subplot(3,1,2)
    # chart=sns.barplot(x='Amount',y='Freq',data=all_amounts)
    # chart.set_xticklabels(chart.get_xticklabels(),rotation=45,fontweight='light',fontsize=4)
    plt.subplot(1,1,1)
    chart = sns.countplot(x='Amount', data=positive_amounts2)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontweight='light', fontsize=6)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def statinfo(v):
    print("mean value of each feature")
    print(np.mean(v,axis = 1))
    print("variance of each feature")
    print(np.var(v,axis = 1))
    print("covariance matrix")
    print(np.cov(v))
    print("correlation matrix")
    print(np.corrcoef(v))
    fig = plt.figure(figsize=(20, 24))
    loc = 1
    for i in range(30):
        ax = fig.add_subplot(6, 5, loc)
        loc += 1
        sns.distplot(v[i], ax=ax)
    # plt.show()

def mean():
    m = np.mean(v1,axis = 1)
    print("mean time:", m[0])
    for i in range(1,29):
        print("mean v"+str(i)+":", m[i])
    print("mean amount:", m[29])

def variance():
    var = np.var(v1,axis = 1)
    print("variance of time:",var[0])
    for i in range(1,29):
        print("variance of v"+str(i)+":",var[i])
    print("variance of amount:",var[29])

d = {"time":0,"amount":29,"class":30,}
for i in range(1,29):
    d["v"+str(i)] = i

def covariance(s,t):
    cov = np.cov(v1)
    print(cov[d[s]][d[t]])

def rho(s,t):
    cor = np.corrcoef(v1)
    print(cor[d[s]][d[t]])

def fraud_mean():
    m = np.mean(v2,axis = 1)
    print("mean time:", m[0])
    for i in range(1,29):
        print("mean v"+str(i)+":", m[i])
    print("mean amount:", m[29])

def fraud_variance():
    var = np.var(v2,axis = 1)
    print("variance of time:",var[0])
    for i in range(1,29):
        print("variance of v"+str(i)+":",var[i])
    print("variance of amount:",var[29])

def fraud_covariance(s,t):
    cov = np.cov(v2)
    print(cov[d[s]][d[t]])

def fraud_rho(s,t):
    cor = np.corrcoef(v2)
    print(cor[d[s]][d[t]])

def frauds_wrt_time_main():
    credit = pd.read_csv("creditcard.csv")
    # time = credit["Time"].to_numpy(dtype=int)

    positive_times = credit[(credit['Class'] == 1)][['Time', 'Class']].drop(columns=['Class']).to_numpy(dtype=int)

    # get_hists(0,172792,positive_times,None,t0)
    a,b = 0, 49*3600
    get_hists_grouped(a, b, 3600, positive_times,bar_width=0.8)
    sns.set()
    # plt.figure()
    # sns.distplot(credit['Time'] / 60 / 60)
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    # plt.hist(time,bins=48) # plotting histogram of all the transmissions in each hour of the 48.

def frauds_wrt_amounts_main():
    sns.set()
    credit_df = pd.read_csv("creditcard.csv")

    all_amounts = credit_df[['Amount']].round(0).astype(int)
    all_amounts['Freq'] = all_amounts.groupby('Amount')['Amount'].transform('count')

    all_amounts2 = all_amounts.drop(all_amounts[all_amounts.Freq<5].index)
    print('all_amounts.shape =',all_amounts2.shape)

    # positive_amounts = credit_df[(credit_df['Class']==1)][['Amount','Class']].drop(columns=['Class']).round(0).astype(int)
    # positive_amounts['Freq'] = positive_amounts.groupby('Amount')['Amount'].transform('count')

    # positive_amounts2 = positive_amounts.drop(positive_amounts[positive_amounts.Freq<2].index)

    # print("positive_amounts2.shape =",positive_amounts2.shape)

    plt.figure()
    # plt.subplot(3,1,1)
    # chart=sns.barplot(x='Amount',y='Freq',data=positive_amounts)
    # chart.set_xticklabels(chart.get_xticklabels(),rotation=45,fontweight='light',fontsize=6)
    #
    # plt.subplot(3,1,2)
    # chart=sns.barplot(x='Amount',y='Freq',data=all_amounts)
    # chart.set_xticklabels(chart.get_xticklabels(),rotation=45,fontweight='light',fontsize=4)
    plt.subplot(1,1,1)
    chart = sns.countplot(x='Amount', data=all_amounts2)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontweight='light', fontsize=6)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def main():
    while True:
        x = input("view statistics: ")
        if x == "mean":
            mean()
        if x == "variance":
            variance()
        if x[:10] == "covariance":
            x = x.split()
            covariance(x[1],x[2])
        if x[:3] == "rho":
            x = x.split()
            rho(x[1],x[2])
        if x == "frauds mean":
            fraud_mean()
        if x == "frauds variance":
            fraud_variance()
        if x[:17] == "frauds covariance":
            x = x.split()
            fraud_covariance(x[2],x[3])
        if x[:10] == "frauds rho":
            x = x.split()
            fraud_rho(x[2],x[3])
        if x == "frauds wrt time":
            frauds_wrt_time_main()
        if x == "frauds wrt amount":
            frauds_wrt_amount_main()

if __name__ == "__main__":
    main()