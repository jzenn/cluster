import numpy as np


def get_similarity_matrix_from_distance_matrix(
    D: np.array, sigma: float = 1.0
) -> np.array:
    """
    construct the similarity matrix from the distance matrix by applying a Gaussian kernel with parameter sigma

    :param D: distance matrix
    :param sigma: parameter sigma used in the Gaussian kernel
    :return: similarity matrix S
    :rtype: np.array

    Example
    ::
        import numpy as np
        from cluster.utils import get_similarity_matrix, get_similarity_matrix_from_distance_matrix

        # some random points
        points = np.random.rand(10, 5)

        distance_matrix = get_similarity_matrix(points, distance=True)
        similarity_matrix = get_similarity_matrix_from_distance_matrix(distance_matrix)

    |



    """
    return np.exp(-(D ** 2) / (2 * sigma ** 2))


def get_distance_matrix_from_similarity_matrix(
    S: np.array, sigma: float = 1.0
) -> np.array:
    """
    construct the distance matrix from the similarity matrix by applying the inverse Gaussian kernel to the similarity
    matrix

    :param S: similarity matrix
    :param sigma: parameter sigma used in the Gaussian kernel
    :return: distance matrix D
    :rtype: np.array

    Example
    ::
        import numpy as np
        from cluster.utils import get_similarity_matrix, get_distance_matrix_from_similarity_matrix

        # some random points
        points = np.random.rand(10, 5)

        similarity_matrix = get_similarity_matrix(points)
        distance_matrix = get_distance_matrix_from_similarity_matrix(similarity_matrix)

    |



    """
    return np.sqrt(-np.log(S) * (2 * sigma ** 2))


def get_similarity_matrix(
    points: np.array,
    sigma: float = 1.0,
    norm: int = None,
    vectorized: bool = True,
    distance: bool = False,
) -> np.array:
    """
    construct the similarity matrix of points by computing distances and applying a Gaussian kernel with
    parameter sigma

    :param points: array of points of size (N x d) where N is the number of points
    :param sigma: parameter for the Gaussian kernel
    :param norm: norm to use for the distance of the points
    :param vectorized: whether to use vectorization (might use more memory initially) or filling the
        similarity matrix iteratively (uses less memory)
    :param distance: whether to return the similarity matrix or the distance matrix
    :return: similarity matrix
    :rtype: np.array

    Example
    ::
        import numpy as np
        from cluster.utils import get_similarity_matrix

        # some random points
        points = np.random.rand(10, 5)

        distance_matrix = get_similarity_matrix(points, distance=True)
        similarity_matrix = get_similarity_matrix(points)

    |



    """
    # expect points to be (N x k)
    N, k = points.shape
    if vectorized:
        M = np.expand_dims(points, 0).repeat(N, 0)
        D = np.linalg.norm((M - np.transpose(M, (1, 0, 2))), axis=2, ord=norm)
    else:
        D = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                p = points[i, :]
                q = points[j, :]
                D[i, j] = np.linalg.norm(p - q, ord=norm)

    if not distance:
        # apply Gaussian kernel to distance matrix to get similarity
        return get_similarity_matrix_from_distance_matrix(D, sigma)
    else:
        return D


def get_random_similarity_matrix(
    n: int = 10,
    d: int = 2,
    sigma: float = 1.0,
    norm: int = None,
    vectorized: bool = True,
    distance: bool = False,
):
    """
    constructs a random similarity matrix with random points of size (n x d)

    :param n: number of points to create
    :param d: number of dimension for each point to create
    :param sigma: parameter for the Gaussian kernel
    :param norm: norm to use for the distance of the points
    :param vectorized: whether to use vectorization (might use more memory initially) or filling the
        similarity matrix iteratively (uses less memory)
    :param distance: whether to return the similarity matrix or the distance matrix
    :return: similarity matrix
    :rtype: np.array

    Example
    ::

        from cluster.utils import get_random_similarity_matrix

        random_similarity_matrix = get_random_similarity_matrix(points, distance=True)

    |



    """
    points = np.random.rand(n, d)
    return get_similarity_matrix(points, sigma, norm, vectorized, distance)
