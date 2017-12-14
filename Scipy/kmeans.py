import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten

np.random.seed(1)

#data initialization
features = array([[np.random.rand() * 3.0, np.random.rand() * 3.0] for i in range(0, 50)])

#computing k-means with 3 clusters
centroids, _ = kmeans(whiten(features), 3)

#assign each sample to a cluster
index, _ = vq(features, centroids)

color = ['ob', 'or', 'og']

fig, ax = plt.subplots()

for i in range(0, 3):
    ax.plot(features[index==i,0], features[index==i,1], color[i])
    ax.plot(centroids[i][0], centroids[i][1], 'sm')

ax.set_title("K-means clustering with scipy")

plt.show()
