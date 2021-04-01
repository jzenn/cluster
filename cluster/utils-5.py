import numpy as np
from cluster.utils import visualize_clusters_2d
from cluster.cluster import SpectralClustering

np.random.seed(42)

N1 = 25
N2 = 75
r1 = 1 / 4
r2 = 3 / 4

x1 = 2 * np.pi * np.random.rand(N1)
x2 = 2 * np.pi * np.random.rand(N2)
r1 = r1 + np.random.rand(N1) * 0.1
r2 = r2 + np.random.rand(N2) * 0.1

points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                    np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

# cluster the dataset
spectral_clustering = SpectralClustering(k=2, graph_param=5)
spectral_clustering.cluster(points=points)
labels = spectral_clustering.get_labels()

# visualize the dataset in 2D
fig, ax = plt.subplots()
ax = visualize_clusters_2d(points, labels, ax, 'SpectralClustering with k = 2')
plt.show()