#!/usr/bin/python

import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import cluster
from numpy import genfromtxt, savetxt

pd.options.mode.chained_assignment = None  # default='warn'
train_df = pd.read_csv("data.csv")

features = np.array(['Body_Parts','Procedures','Average_Subsidised_Fees'])
input_values = train_df['Average_Subsidised_Fees']

#setup data
index = 1
data = [];
for x in train_df['Average_Subsidised_Fees']:
	data.append([index, x])
	index = index + 1

print("\nData: \n")
print(data)

#K-means clustering
model_var = KMeans(n_clusters=3).fit(data)
centroids = model_var.cluster_centers_
print("\nCentroids: \n")
print(centroids)

#Plot the clustering
plt.plot(range(1,66), train_df['Average_Subsidised_Fees'],'.')
plt.plot(centroids[:,0], centroids[:,1],'o')
plt.axis([0, 70, 0, 10000])
plt.show()

print("\nDone.\n")


