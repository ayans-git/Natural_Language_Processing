#K-Means is the most popular Clustering algorithm
#It works on unlabled data (unsupervised learning)
#K-Means is primarily used clustering problems like Market Segmentation
#It is an iterative algorithm with 2 primary steps:
  #Step 1: Cluster Assignment
  #Step 2: Move Centroid
#Key consideration: Choosing value of K
  #Most common and effective way is trial by hand and visualization
  #Another popular way is Elbow Method. However, in real world data a clear "elbow" might be difficult to obtain
    #In "Elbow Method" X-axis: K (number of Clusters) and Y-axis: Cost Function (J)
#########################################################################################################################
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#use the blob function that produces a predefined number (n_features) of separate point clouds
n=1500
X, y = make_blobs(n_samples = n, n_features = 3, random_state = 42)
plt.scatter(X[:,0], X[:,1])

# Run the KMeans command from the sklearn.cluster package. 
# It only requires the number of clusters (n_clusters)
kmeans = KMeans(n_clusters = 3, random_state = 0)                   
kmeans.fit(X)                  
plt.scatter(X[:, 0], X[:, 1],c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'yellow')
