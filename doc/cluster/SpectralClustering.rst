Spectral Clustering
===================

``cluster.cluster`` provides an interface for k-Means Clustering and Spectral Clustering.

.. autoclass:: cluster.cluster.SpectralClustering
    :members:


Example
-------
::

  import numpy as np
  from scipy.stats import multivariate_normal
  from cluster.cluster import SpectralClustering
  from cluster.utils import visualize_clusters_2d, visualize_graph_2d
  # matplotlib only for visualization
  import matplotlib.pyplot as plt

  # create the dataset
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

  # visualize the graph
  fig, ax = plt.subplots()
  visualize_graph_2d(points, spectral_clustering.get_graph(), ax, 'kNN graph with 5 neighbors')
  plt.show()

.. plot::

  import numpy as np
  from scipy.stats import multivariate_normal
  from cluster.cluster import SpectralClustering
  from cluster.utils import visualize_clusters_2d, visualize_graph_2d
  # matplotlib only for visualization
  import matplotlib.pyplot as plt

  # create the dataset
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

  import numpy as np
  from scipy.stats import multivariate_normal
  from cluster.cluster import SpectralClustering
  from cluster.utils import visualize_clusters_2d, visualize_graph_2d
  # matplotlib only for visualization
  import matplotlib.pyplot as plt

  # create the dataset
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

  # visualize the graph
  fig, ax = plt.subplots()
  visualize_graph_2d(points, spectral_clustering.get_graph(), ax, 'kNN graph with 5 neighbors')
  plt.show()
