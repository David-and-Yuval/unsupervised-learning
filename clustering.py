#import numpy as np
#import pandas as pd
#from sklearn.cluster import KMeans

#df = pd.read_csv('creditcard.csv')
#print('dataframe shape=',df.shape)

#time_vec = df['Time']
#print('time_vec: shape={}, type={}'.format(time_vec.shape,type(time_vec)))
#vecs = df[['V'+str(num) for num in range(1,29)]]
#print('vecs: shape={}, type={}'.format(vecs.shape,type(vecs)))
#amounts_vec = df['Amount']
#print('amounts_vec: shape={}, type={}'.format(amounts_vec.shape,type(amounts_vec)))
#classes = df['Class']
#print('classes: shape={}, type={}'.format(classes.shape,type(classes)))

#select_positives = df.loc[df['Class']==1]
#print('amount of positive values:',select_positives.shape)

#vecs = vecs.to_numpy()
#print('vecs: shape={}, type={}'.format(vecs.shape,type(vecs)))

from stats import *

def cutoff(M,th):
    lab = [0]*len(M)
    for i in range(len(M)):
        if M[i][1] > th:
            lab[i] = 1
    return lab

def sample_vecs(df,ratio=1):
    X1 = df[df['Class'] == 1]
    assert len(X1)==492
    X2 = df[df['Class'] == 0].sample(int(len(X1)*ratio))
    X = pd.concat([X1, X2], axis=0)
    X = X.sample(frac=1).reset_index(drop=True)
    sampled_labels = X[['Class']].to_numpy().reshape(-1)
    X = X.drop(['Time','Amount','Class'], axis=1).to_numpy()
    return X, sampled_labels

def cluster(vecs, gt):


    kmeans = KMeans(n_clusters=2).fit(vecs)
    kmeans_labels = kmeans.labels_
    if sum(kmeans_labels) > len(kmeans_labels)/2:
        kmeans_labels = 1 - kmeans_labels
    kmeans_centers = kmeans.cluster_centers_
#    scatter(vecs[:,13], vecs[:,16], ax=axes[0], hue=kmeans_labels)
    # scatter(kmeans_centers[:,14], kmeans_centers[:,17], ax=axes[0],marker="s",s=100)

    fcm = FCM(n_clusters=2, m=1.1).fit(vecs)
    fcm_centers = fcm.centers
    fcm_labels  = cutoff(fcm.u,0.6)
    #fcm_labels = fcm.u.argmax(axis = 1)
    if sum(fcm_labels) > len(fcm_labels)/2:
        fcm_labels = 1 - np.array(fcm_labels)
    # print('fcm_centers:\n',fcm_centers)
    # print('fcm_labels:\n',fcm_labels)
#    scatter(vecs[:,13], vecs[:,16], ax=axes[1], hue=fcm_labels)
    # scatter(fcm_centers[:,14], fcm_centers[:,17], ax=axes[1],marker="s",s=100)

    gmm = GaussianMixture(n_components=2).fit(vecs)
    gmm_labels = gmm.predict(vecs)
    if sum(gmm_labels) > len(gmm_labels)/2:
        gmm_labels = 1 - gmm_labels

    db_labels = dbscan(vecs,3,1)

    print("kmeans")
    print(randomize(kmeans_labels,gt))
    print("fcm")
    print(randomize(fcm_labels,gt))
    print("gmm")
    print(randomize(gmm_labels,gt))
    print("dbscan")
    print(randomize(db_labels,gt))

def dbscan(vecs,samples,eps):
    labels = DBSCAN(min_samples = samples,eps = eps).fit_predict(vecs)
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

#    plt.show()

def dbratio():
    table2 = pd.DataFrame(columns=['ratio', 'samples', 'eps', 'precision', 'recall', 'f-1 score'])
    for ratio in [1,1.2,1.5,1.8]:
        for samples in range(10,22):
            for eps in [3.6,4,4.4,4.8,5.2,5.6,6,6.4]:
                vecs_sampled, labels_sampled = sample_vecs(credit, ratio)
                labels = dbscan(vecs_sampled,samples,eps)
                metrics = precision_recall_fscore_support(labels_sampled, labels, average='binary')
                    # print('ratio={}, m={}: precision: {}, recall: {}, f-1 score: {}'.format(axes[i].get_title(), m, *metrics[:-1]))
                dftemp = pd.DataFrame([[ratio, samples, eps, metrics[0], metrics[1], metrics[2]]],
                                      columns=['ratio', 'samples', 'eps', 'precision', 'recall', 'f-1 score'])
                table2 = table2.append(dftemp, ignore_index=True)
    filepath = "db.xlsx"
    table2.to_excel(filepath)

def randomize(pred,gt):
    bm = f1_score(pred,gt)
    print(bm)
    s = sum(gt)
    l = len(gt)
    res = 0
    for _ in range(100):
        r = random.sample(range(l),s)
        new = [0 for _ in range(l)]
        for i in range(s):
            new[r[i]] = 1
        f1 = f1_score(pred,new)
        if f1 < bm:
            res += 1
    return res/100


if __name__ == "__main__":
    print("pca features")
    cluster(vecs,gt)
    print("features+amount")
    cluster(vecs_a,gt)
    print("normalized")
    cluster(vecs_z,gt)
    print("significant 2")
    cluster(vecs_2,gt)
    print("high amount")
    cluster(vecs_h,gt_h)
    # print(precision_recall_fscore_support(labels,gt,average = "binary"))
