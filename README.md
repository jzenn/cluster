# The ``cluster`` package

## Documentation

This is a lightweight clustering package implementing Spectral Clustering as well as k-Means Clustering.
There is some documentation available, that can be found [*here*](http://jzenn.github.io/cluster).

Installation
------------

You can simply install the package with ``pip`` directly from the repository

```
pip install git+https://github.com/jzenn/cluster
```

The package requires at least *Python 3.6* to work properly. Additonal dependencies are:

* numpy
* scipy
* matplotlib (just for visualizations, not necessarily needed)

API Reference
-------------

* ``cluster.cluster`` provides the functionality to cluster a dataset.
* ``cluster.utils`` provides methods to construct similarity/distance matrices and differnt graphs from them.
* ``cluster.graph`` is a lightweight graph implementation.

A documentation with greater depth can be found [*here*](http://jzenn.github.io/cluster).

## Roadmap

- implement regularized spectral clustering
