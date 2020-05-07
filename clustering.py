# from script import *
from stats import *
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from fcmeans import FCM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
# import datetime
from seaborn import scatterplot as scatter
import warnings
from scipy.spatial import distance_matrix as dist
import random
random.seed(9)


def cutoff(M,thresh):
    """idea: put this method inside the fcmeans class"""
    labels = [0]*len(M)
    for i in range(len(M)):
        if M[i][1] > thresh:
            labels[i] = 1
    return labels

def sample_vecs(df,ratio=1):
    X1 = df[df['Class'] == 1]
    assert len(X1)==492
    random.seed(9)
    X2 = df[df['Class'] == 0].sample(int(len(X1)*ratio),random_state = 900)
    X = pd.concat([X1, X2], axis=0)
    X = X.sample(frac=1,random_state = 0).reset_index(drop=True)
    sampled_labels = X[['Class']].to_numpy().reshape(-1)
    X = X.drop(['Time','Amount','Class'], axis=1).to_numpy()
    return X, sampled_labels

def dbscan_func(vecs,samples,eps):
    labels = DBSCAN(min_samples = samples,eps = eps).fit_predict(vecs)
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

def kmeans():
    X,s = sample_vecs(credit, ratio = 100)
    kmeans = KMeans(n_clusters=2, random_state = 0).fit(X)
    kmeans_labels = kmeans.labels_
    if sum(kmeans_labels) > len(kmeans_labels)/2:
        kmeans_labels = 1 - kmeans_labels
    f, axes = plt.subplots(1, 2)
    scatter(X[:,0], X[:,1], ax=axes[0],hue=s)
    scatter(X[:,0], X[:,1], ax=axes[1], hue=kmeans_labels)
    results = precision_recall_fscore_support(kmeans_labels,s,average = "binary")
    print("precision:",results[0])
    print("recall:",results[1])
    print("f1:",results[2])
    plt.show()

def fcm():
    X,s = sample_vecs(credit, ratio = 7)
    fcm = FCM(n_clusters=2, m=2.5).fit(X)
    fcm_labels  = cutoff(fcm.u,0.6)
    #fcm_labels = fcm.u.argmax(axis = 1)
    if sum(fcm_labels) > len(fcm_labels)/2:
        fcm_labels = 1 - np.array(fcm_labels)
    f, axes = plt.subplots(1, 2)
    scatter(X[:, 0], X[:, 1], ax=axes[0], hue=s)
    scatter(X[:, 0], X[:, 1], ax=axes[1], hue=fcm_labels)
    results = precision_recall_fscore_support(fcm_labels, s,average = "binary")
    print("precision:", results[0])
    print("recall:", results[1])
    print("f1:", results[2])
    plt.show()

def gmm():
    X, s = sample_vecs(credit, ratio=2)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    gmm_labels = gmm.predict(X)
    if sum(gmm_labels) > len(gmm_labels) / 2:
        gmm_labels = 1 - gmm_labels
    f, axes = plt.subplots(1, 2)
    scatter(X[:, 0], X[:, 1], ax=axes[0], hue=s)
    scatter(X[:, 0], X[:, 1], ax=axes[1], hue=gmm_labels)
    results = precision_recall_fscore_support(gmm_labels, s,average = "binary")
    print("precision:", results[0])
    print("recall:", results[1])
    print("f1:", results[2])
    plt.show()

def dbscan():
    X, s = sample_vecs(credit)
    db = dbscan_func(X,11,4.4)
    f, axes = plt.subplots(1, 2)
    scatter(X[:, 0], X[:, 1], ax=axes[0], hue=s)
    scatter(X[:, 0], X[:, 1], ax=axes[1], hue=db)
    results = precision_recall_fscore_support(db, s,average = "binary")
    print("precision:", results[0])
    print("recall:", results[1])
    print("f1:", results[2])
    plt.show()

def spectral():
    warnings.filterwarnings('ignore')
    X, s = sample_vecs(credit,ratio = 3)
    distances = dist(X,X)
    spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
    spec_labels = spec.fit(distances).labels_
    f, axes = plt.subplots(1, 2)
    scatter(X[:, 0], X[:, 1], ax=axes[0], hue=s)
    scatter(X[:, 0], X[:, 1], ax=axes[1], hue=spec_labels)
    results = precision_recall_fscore_support(spec_labels, s,average = "binary")
    print("precision:", results[0])
    print("recall:", results[1])
    print("f1:", results[2])
    plt.show()
    return spec_labels

def p_values():
    vecs_kmeans, gt_kmeans = sample_vecs(credit,ratio = 100)
    vecs_fcm, gt_fcm = sample_vecs(credit,ratio = 7)
    vecs_gmm, gt_gmm = sample_vecs(credit,ratio = 2)
    vecs_spectral, gt_spectral = sample_vecs(credit,ratio = 3)
    vecs_dbscan, gt_dbscan = sample_vecs(credit,ratio = 1)

    kmeans = KMeans(n_clusters=2, random_state = 0).fit(vecs_kmeans)
    kmeans_labels = kmeans.labels_
    if sum(kmeans_labels) > len(kmeans_labels)/2:
        kmeans_labels = 1 - kmeans_labels
    kmeans_centers = kmeans.cluster_centers_
#    scatter(vecs[:,13], vecs[:,16], ax=axes[0], hue=kmeans_labels)
    # scatter(kmeans_centers[:,14], kmeans_centers[:,17], ax=axes[0],marker="s",s=100)

    fcm = FCM(n_clusters=2, m=1.1).fit(vecs_fcm)
    fcm_centers = fcm.centers
    fcm_labels  = cutoff(fcm.u,0.6)
    #fcm_labels = fcm.u.argmax(axis = 1)
    if sum(fcm_labels) > len(fcm_labels)/2:
        fcm_labels = 1 - np.array(fcm_labels)
    # print('fcm_centers:\n',fcm_centers)
    # print('fcm_labels:\n',fcm_labels)
#    scatter(vecs[:,13], vecs[:,16], ax=axes[1], hue=fcm_labels)
    # scatter(fcm_centers[:,14], fcm_centers[:,17], ax=axes[1],marker="s",s=100)

    gmm = GaussianMixture(n_components=2, random_state = 0).fit(vecs_gmm)
    gmm_labels = gmm.predict(vecs_gmm)
    if sum(gmm_labels) > len(gmm_labels)/2:
        gmm_labels = 1 - gmm_labels


    warnings.filterwarnings('ignore')
    spectral_labels = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0).fit(dist(vecs_spectral,vecs_spectral)).labels_
    # spec_labels = spec.fit(distances).labels_

    # spectral_labels = spectral()

    db_labels = dbscan_func(vecs_dbscan,11,4.4)

    print("kmeans")
    print(randomize(kmeans_labels,gt_kmeans))
    print("fcm")
    print(randomize(fcm_labels,gt_fcm))
    print("gmm")
    print(randomize(gmm_labels,gt_gmm))
    print("spectral")
    print(randomize(spectral_labels, gt_spectral))
    print("dbscan")
    print(randomize(db_labels,gt_dbscan))

def randomize(pred,gt):
    bm = f1_score(gt,pred)
    print(bm)
    ext = [1,0]
    s = sum(gt)
    l = len(gt)
    res = 0
    for it in range(100):
        random.seed(it)
        r = random.sample(range(l),s)
        new = [0 for _ in range(l)]
        for i in range(s):
            new[r[i]] = 1
        f1 = f1_score(new,pred)
        if f1 < bm:
            res += 1
        if f1 < ext[0]:
            ext[0] = f1
        if f1 > ext[1]:
            ext[1] = f1
    return ext, res/100



if __name__ == "__main__":
    # david_clustering()
    # main_spectral()
    # yuval_clustering()
    options = {'kmeans':'K-Means', 'fuzzy c means':'Fuzzy C-Means', 'gmm':'GMM',
               'spectral clustering':'Spectral Clustering', 'dbscan':'DBSCAN', 'p-value':'p-values (of all the algorithms)'}
    print('Algorithms Menu:')
    for k,v in options.items(): print("type '{}' for {}".format(k,v))
    choice = input('Choose one of the options\n')
    if choice=='kmeans':
        kmeans()
    elif choice=='fuzzy c means':
        fcm()
    elif choice=='gmm':
        gmm()
    elif choice=='spectral clustering':
        spectral()
    elif choice=='dbscan':
        dbscan()
    elif choice=='p-value':
        p_values()
    else:
        raise Exception("Invalid choice!")
    # print('-------------------')
