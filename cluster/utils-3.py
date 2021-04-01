import numpy as np
from cluster.utils import kNN_graph_from_similarity_matrix, visualize_graph_2d, get_similarity_matrix

np.random.seed(42)

N1 = 10
N2 = 10
r1 = 1 / 4
r2 = 3 / 4

x1 = 2 * np.pi * np.random.rand(N1)
x2 = 2 * np.pi * np.random.rand(N2)
r1 = r1 + np.random.rand(N1) * 0.1
r2 = r2 + np.random.rand(N2) * 0.1

points = np.vstack([np.concatenate([r1 * np.cos(x1), r2 * np.cos(x2)]),
                    np.concatenate([r1 * np.sin(x1), r2 * np.sin(x2)])]).T

fig, ax = plt.subplots()
visualize_graph_2d(points, kNN_graph_from_similarity_matrix(get_similarity_matrix(points), k=3), ax,
'3-NN graph')
plt.show()