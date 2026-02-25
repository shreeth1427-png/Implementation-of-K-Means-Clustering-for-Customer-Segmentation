# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select the required features (Annual Income and Spending Score).

2. Choose the number of clusters (K) using the Elbow Method.

3. Initialize centroids and assign data points to the nearest centroid based on minimum distance.

4. Update the centroids by calculating the mean of each cluster and repeat the process until the centroids stop changing. 


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: TEJASHREE M
RegisterNumber: 212225220115 
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")

print(dataset.head())

X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids')

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

## Output:
<img width="1044" height="588" alt="Screenshot 2026-02-25 111653" src="https://github.com/user-attachments/assets/3a072337-6153-429b-8d00-368a60119bf3" />
<img width="747" height="147" alt="image" src="https://github.com/user-attachments/assets/a1acbf67-8d14-4f8e-8aca-17898f9ba5a2" />
<img width="761" height="567" alt="Screenshot 2026-02-25 110237" src="https://github.com/user-attachments/assets/df3911f7-25b1-4b97-8a02-3b15694032f2" />
<img width="786" height="567" alt="Screenshot 2026-02-25 110345" src="https://github.com/user-attachments/assets/fce58f11-19b5-4b94-acd1-a55351311bd1" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
