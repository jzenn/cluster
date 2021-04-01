.. cluster documentation master file

Documentation of ``cluster``
============================

This is a lightweight clustering package implementing Spectral Clustering as well as k-Means Clustering.
For an overview over the topic
the following reference by Prof. Ulrike von Luxburg might be a good start.

*Ulrike von Luxburg (2007). A tutorial on spectral clustering. Stat. Comput., 17(4), 395–416.*

Installation
------------

You can simply install the package with ``pip`` directly from the repository. ::

  pip install git+https://github.com/jzenn/cluster

The package requires at least *Python 3.6* to work properly. Additonal dependencies are:

* numpy
* scipy
* matplotlib (just for visualizations, not necessarily needed)

API Reference
-------------

* ``cluster.cluster`` provides the functionality to cluster a dataset.
* ``cluster.utils`` provides methods to construct similarity/distance matrices and differnt graphs from them.
* ``cluster.graph`` is a lightweight graph implementation.

.. toctree::
   :maxdepth: 1
   :caption: Clustering

   cluster/SpectralClustering
   cluster/KMeansClustering
   cluster/utils

.. toctree::
   :maxdepth: 1
   :caption: Graph:

   graph/Graph
   graph/Node
   graph/Edge
