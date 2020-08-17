# Clustering

The Python `cluster` package implements Spectral 
Clustering as well as k-Means Clustering.
It implements the functionality of both algorithms from scratch.
This gives the possibility to fine-tune every parameter the algorithm has.  

*k-Means*
on the circle dataset and on a mixture of Gaussians.
![Circles and Mixture kMeans Clustering](https://raw.githubusercontent.com/jzenn/cluster/master/assets/kMeans_clustering.png)


*Spectral Clustering*
- a similarity matrix can be obtained from points
- four different graph types can be used for the algorithm
    - eps-graph
    - kNN-graph
    - mutual kNN graph
    - fully connected graph
- the graph Laplacian is computed
    - in its unnormalized version
    - in its symmetric and normalized version
- the eigenvectors and eigenvalues of the graph Laplacian are computed
    - with an implementation of the power method
    - using `scipy`'s `eigh` and `eigsh` methods
- the eigenvectors are clustered with k-Means

Here are some clustering examples with parameter combinations.

![Circles Spectral Custering](https://raw.githubusercontent.com/jzenn/cluster/master/assets/circles_spectral_clustering.png)
![Mixture Spectral Custering](https://raw.githubusercontent.com/jzenn/cluster/master/assets/mixture_spectral_clustering.png)

## Installation

You can simply install the provided `.whl` file from the `dist` 
folder with `pip`. Please note that you need to have `> Python 3.6` 
installed.
 
```
pip install dist/cluster_jzenn-0.0.1-py3-none-any.whl
```

Additionally, the packages `numpy`, `matplotlib` and `scipy`.

---
## Modules 
All modules/classes are summarized below.

### `cluster.cluster`
`cluster.cluster` provides interfaces for the `KMeansClustering` and `SpectralClustering`
as well as `visualize_clusters_2d`, `visualize_graph_clusters_2d`.

*KMeansClustering*
```
from cluster.cluster import KMeansClustering, visualize_clusters_2d

k = 2
kmeans_clustering = KMeansClustering(k=k, max_iter=1000, seed=0, eps=1e-10, norm=None, rep=5)
kmeans_clustering.cluster(points)

fig, ax = plt.subplots(figsize=(15, 10))
ax = visualize_clusters_2d(points, kmeans_clustering.get_labels(), ax, f'kMeans with k = {k}')
plt.show()
```

*Spectral Clustering*
```
from cluster.cluster import SpectralClustering, visualize_graph_clusters_2d

k = 2
spectral_clustering = SpectralClustering(
    k, max_iter=1000, seed=0, eps=1e-10, norm=None, device='cpu', 
    sparse=True, graph='kNN', graph_param=10, normalized=True, 
    vectorize_similarity_matrix=True, use_scipy_eigh=False)
spectral_clustering.cluster(points)

fix, ax = plt.subplots(1, 2, figsize=(15, 5))
titles = [f'Spectral Clustering with k = {k}, kNN, ', f'Spectral Clustering with k = {k}, clusters']

ax = visualize_graph_clusters_2d(points, spectral_clustering, ax, titles=titles, annotate=False)
plt.show()
```

### `cluster.graph`
`cluster.graph` provides interfaces for `Graph`, `Node` and `Edge`. There are also 
functions to create graphs from similarity matrices: `eps_graph_from_similarity_matrix, 
fc_graph_from_similarity_matrix, kNN_graph_from_similarity_matrix, 
mkNN_graph_from_similarity_matrix`. Further, there is the possiblity to visualize a two-
dimensional graph with `visualize_graph_2d`.

*eps-graph*
```
from cluster.graph import eps_graph_from_similarity_matrix

eps = 0.3
graph = eps_graph_from_similarity_matrix(S, eps)

fig, ax = plt.subplots(figsize=(15, 10))
ax = visualize_graph_2d(points, graph, f'eps-graph, eps={eps}')
plt.show()
```

*fc-graph*
```
from cluster.graph import fc_graph_from_similarity_matrix

graph = fc_graph_from_similarity_matrix(S)

fig, ax = plt.subplots(figsize=(15, 10))
ax = visualize_graph_2d(points, graph, 'fc-graph')
plt.show()
```

*mkNN-graph*
```
from cluster.graph import mkNN_graph_from_similarity_matrix

k = 3
graph = mkNN_graph_from_similarity_matrix(S, k)

fig, ax = plt.subplots(figsize=(15, 10))
ax = visualize_graph_2d(points, graph, f'mkNN-graph, k={k}')
plt.show()
```

*kNN-graph*
```
from cluster.graph import kNN_graph_from_similarity_matrix

k = 3
graph = kNN_graph_from_similarity_matrix(S, k)

fig, ax = plt.subplots(figsize=(15, 10))
ax = visualize_graph_2d(points, graph, f'kNN-graph, k={k}')
plt.show()
```

### `cluster.linalg`
`cluster.linalg` provides an interface to compute the largest eigenvalue (in its 
absolute value) of a diagonalizable matrix with `power`. Additionally, 
if the matrix is symmetric, you can compute the k largest eigenvalues and eigenvectors 
with `get_last_k_eig`. Furthermore, if the matrix is symmetric and positive-semidefinite,
you can compute the k smallest eigenvalues and eigenvectors with `get_first_k_eig`.

*power method*
```
from cluster.linalg import power

v, lamb = power(A, eps=1e-15, max_iter=100000, norm=None)
```

*last k eigenvectors*
```
from cluster.linalg import get_last_k_eig
from scipy.sparse import csr_matrix

vs, lambs = get_last_k_eig(csr_matrix(L), k=4, sparse=True)
```

*first k eigenvectors*
```
from cluster.linalg import get_first_k_eig
from scipy.sparse import csr_matrix

vs, lambs = get_first_k_eig(csr_matrix(L), k=4, sparse=True)
```

### `cluster.stats`
`cluster.stats` provides an interface to create multivariate normal distributions 
with `MultivariateNormal` and sample from them (and visualize a two dimensional one with 
`visualize_multivariate_2d`). Also, you can create mixture normal distributions 
with `MixtureNormal` and sample from them (and visualize a two dimensional one with 
`visualize_mixture_2d`).

```
import 

n = 250

mu_1 = np.array([0, -0.5])
mu_2 = np.array([-0.5, 0.25])
mu_3 = np.array([0.75, 0.])

sigma_1 = np.diag(np.array([0.01, 0.01]))
sigma_2 = np.diag(np.array([0.05, 0.01]))
sigma_3 = np.diag(np.array([0.01, 0.05]))

mix = MixtureNormal(multivariates=[MultivariateNormal(mean=mu_1, covariance=sigma_1), 
                                   MultivariateNormal(mean=mu_2, covariance=sigma_2),
                                   MultivariateNormal(mean=mu_3, covariance=sigma_3),], 
                    weights=[1/3] * 3)

fig, ax = plt.subplots(figsize=(15, 10))
ax, points = visualize_mixture_2d(mix, n, ax, 'Mixture with three Normal Distributions')
plt.show()
```

## Roadmap

- implement regularized spectral clustering
