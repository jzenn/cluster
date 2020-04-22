import warnings
import numpy as np
import numpy.linalg as lina

from scipy.sparse import csr_matrix


def reighleigh_quotient(A, v):
    # v^T Av / |v| 
    return v.dot(A.dot(v)) / v.dot(v)


def power(A, eps=1e-15, max_iter=100000, norm=None):
    v = np.random.rand(A.shape[1])
    gap = np.infty
    i = 0

    while gap > eps and i < max_iter:
        # power method
        q = A.dot(v)
        q = q / lina.norm(q, ord=norm)

        # stop criterion
        gap = lina.norm(q - v, ord=norm)
        v = q

        i += 1

    if i == max_iter:
        warnings.warn(f'reached maximum number of iterations with {max_iter}')

    return v, reighleigh_quotient(A, v)


def get_last_k_eig(A, k, sparse=False, norm=None):
    n = A.shape[0]

    # note that A needs to be symmetric
    assert (np.sum(A != A.T) == 0)

    # exactly n eigenvectors
    if k > n: k = n

    eig_vals = np.zeros(k)
    eig_vecs = np.zeros([k, n])

    eig_vec, eig_val = power(A)

    eig_vals[0] = eig_val
    eig_vecs[0, :] = eig_vec

    for i in range(1, k):
        eig_val, eig_vec = eig_vals[i - 1], eig_vecs[i - 1, :].T
        if sparse:
            A = A - eig_val / np.power(lina.norm(eig_vec, ord=norm), 2) * csr_matrix(np.outer(eig_vec, eig_vec))
        else:
            A = A - eig_val / np.power(lina.norm(eig_vec, ord=norm), 2) * np.outer(eig_vec, eig_vec)

        eig_vec, eig_val = power(A)

        eig_vals[i] = eig_val
        eig_vecs[i, :] = eig_vec

    return eig_vecs, eig_vals


def get_first_k_eig(A, k, sparse=False, norm=None):
    n = A.shape[0]

    # note that A needs to be symmetric and positive semi-definite
    # checking ONLY symmetry for efficiency
    assert (np.sum(A != A.T) == 0)

    # exactly n eigenvectors
    if k > n: k = n

    eig_vals = np.zeros(k)
    eig_vecs = np.zeros([k, n])

    # greatest eigenvalue to transform other eigenvalues
    _, last_eig_val = power(A)

    if sparse:
        A = A - last_eig_val * csr_matrix(np.eye(A.shape[1]))
    else:
        A = A - last_eig_val * np.eye(A.shape[1])

    eig_vec, eig_val = power(A)

    eig_vals[0] = eig_val
    eig_vecs[0, :] = eig_vec

    for i in range(1, k):
        eig_val, eig_vec = eig_vals[i - 1], eig_vecs[i - 1, :].T

        if sparse:
            A = A - eig_val / np.power(lina.norm(eig_vec, ord=norm), 2) * csr_matrix(np.outer(eig_vec, eig_vec))
        else:
            A = A - eig_val / np.power(lina.norm(eig_vec, ord=norm), 2) * np.outer(eig_vec, eig_vec)

        eig_vec, eig_val = power(A)

        eig_vals[i] = eig_val
        eig_vecs[i, :] = eig_vec

    return eig_vecs, eig_vals + last_eig_val
