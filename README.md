# unsupervised
The code consists of two main function files: stats.py and clustering.py
stats.py lets the user view statistics on the data: mean, variance, covariance and correlation coeffieicnt, between points on the whole data and within the fraud data.
For the mean, type "mean".
For the variance, type "variance".
For the covariance between two features, write "covariance" and then the features' names.
For instance, if you wish to see the covariance between amount and v22, write "covariance amount v22".
For the correlation coefficient between two features, write "rho" and then the features's names.
To see any of those on the fraud data alone, write the word "frauds" in the beginning of the request.
For instance, if you wish to see the covariance between amount and v22 on the fraud data, write "frauds covariance amount v22".

clustering.py lets the user view the results of our five clustering algorithms on the data. We give plots and evaluations for the optimized parameters.
For kmeans, type "kmeans".
For fuzzy c means, type "fuzzy c means".
For GMM, type "gmm".
For DBSCAN, type "dbscan".
For spectral clustering, type "spectral clustering".

Be sure to use lower-case letters only in all your requests.

Besides those two main function files we have two auxilliary functions: one is called frauds_wrt_time.py, which creates a histogram of the frauds as a function of the time. The other is called fcmeans.py, and it is an implementation of the fuzzy c means algorithm.
