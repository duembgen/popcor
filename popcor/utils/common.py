"""Common utilities for symmetric matrix vectorization and sparsity handling."""

import itertools
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def upper_triangular(p: np.ndarray) -> np.ndarray:
    """Return the vectorized upper-triangular part of the outer product of p."""
    return np.outer(p, p)[np.triu_indices(len(p))]


def diag_indices(n: int) -> np.ndarray:
    """Return indices of diagonal elements in the vectorized upper-triangular matrix."""
    z = np.empty((n, n))
    z[np.triu_indices(n)] = range(int(n * (n + 1) / 2))
    return np.diag(z).astype(int)


def get_aggregate_sparsity(matrix_list_sparse: Sequence[sp.spmatrix]) -> sp.csr_matrix:
    """Aggregate sparsity pattern from a list of sparse matrices."""
    agg_ii: List[int] = []
    agg_jj: List[int] = []
    for A_sparse in matrix_list_sparse:
        assert isinstance(A_sparse, sp.spmatrix)
        ii, jj = A_sparse.nonzero()  # type: ignore
        agg_ii += list(ii)
        agg_jj += list(jj)
    return sp.csr_matrix(([1.0] * len(agg_ii), (agg_ii, agg_jj)), A_sparse.shape)


def unravel_multi_index_triu(
    flat_indices: Sequence[int], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat indices to (i, j) indices for upper-triangular part of a matrix."""
    i_upper: List[int] = []
    j_upper: List[int] = []
    cutoffs = np.cumsum(list(range(1, shape[0] + 1))[::-1])
    for idx in flat_indices:
        i = np.where(idx < cutoffs)[0][0]
        if i == 0:
            j = idx
        else:
            j = idx - cutoffs[i - 1] + i
        i_upper.append(i)
        j_upper.append(j)
    return np.array(i_upper), np.array(j_upper)


def ravel_multi_index_triu(
    index_tuple: Tuple[np.ndarray, np.ndarray], shape: Tuple[int, int]
) -> List[int]:
    """Convert (i, j) indices to flat indices for upper-triangular part of a matrix."""
    ii, jj = index_tuple
    triu_mask = jj >= ii
    i_upper = ii[triu_mask]
    j_upper = jj[triu_mask]
    flat_indices: List[int] = []
    for i, j in zip(i_upper, j_upper):
        idx = np.sum(range(shape[0] - i, shape[0])) + j
        flat_indices.append(idx)
    return flat_indices


def create_symmetric(
    vec: np.ndarray | sp.csr_matrix | sp.csc_matrix,
    eps_sparse: float,
    correct: bool = False,
    sparse: bool = False,
) -> np.ndarray | sp.csr_array:
    """Create a symmetric matrix from the vectorized upper-triangular elements."""

    def get_dim_x(len_vec: int) -> int:
        return int(0.5 * (-1 + np.sqrt(1 + 8 * len_vec)))

    if isinstance(vec, np.ndarray):
        len_vec = len(vec)
        dim_x = get_dim_x(len_vec)
        triu = np.triu_indices(n=dim_x)
        mask = np.abs(vec) > eps_sparse
        triu_i_nnz = triu[0][mask]
        triu_j_nnz = triu[1][mask]
        vec_nnz = vec[mask]
    else:
        len_vec = vec.shape[1]
        dim_x = get_dim_x(len_vec)
        vec.data[np.abs(vec.data) < eps_sparse] = 0
        vec.eliminate_zeros()
        ii, jj = vec.nonzero()
        triu_i_nnz, triu_j_nnz = unravel_multi_index_triu(jj, (dim_x, dim_x))
        vec_nnz = np.array(vec[ii, jj]).flatten()

    if sparse:
        offdiag = triu_i_nnz != triu_j_nnz
        diag = triu_i_nnz == triu_j_nnz
        triu_i = triu_i_nnz[offdiag]
        triu_j = triu_j_nnz[offdiag]
        diag_i = triu_i_nnz[diag]
        if correct:
            vec_nnz_off = vec_nnz[offdiag] / np.sqrt(2)
        else:
            vec_nnz_off = vec_nnz[offdiag]
        vec_nnz_diag = vec_nnz[diag]
        Ai = sp.csr_array(
            (
                np.r_[vec_nnz_diag, vec_nnz_off, vec_nnz_off],
                (np.r_[diag_i, triu_i, triu_j], np.r_[diag_i, triu_j, triu_i]),
            ),
            (dim_x, dim_x),
            dtype=float,
        )
    else:
        Ai = np.zeros((dim_x, dim_x))
        if correct:
            Ai[triu_i_nnz, triu_j_nnz] = vec_nnz / np.sqrt(2)
            Ai[triu_j_nnz, triu_i_nnz] = vec_nnz / np.sqrt(2)
            Ai[range(dim_x), range(dim_x)] *= np.sqrt(2)
        else:
            Ai[triu_i_nnz, triu_j_nnz] = vec_nnz
            Ai[triu_j_nnz, triu_i_nnz] = vec_nnz
    return Ai


def get_vec(
    mat: np.ndarray | sp.csc_matrix | sp.csr_matrix,
    correct: bool = True,
    sparse: bool = False,
) -> np.ndarray | sp.csr_matrix:
    """Convert NxN symmetric matrix to vectorized upper-triangular form preserving inner product."""
    from copy import deepcopy

    mat = deepcopy(mat)
    if correct:
        if isinstance(mat, (sp.csc_matrix, sp.csr_matrix)):
            ii, jj = mat.nonzero()
            mat[ii, jj] *= np.sqrt(2.0)
            diag = ii == jj
            mat[ii[diag], jj[diag]] /= np.sqrt(2)  # type: ignore
        else:
            mat *= np.sqrt(2.0)
            mat[range(mat.shape[0]), range(mat.shape[0])] /= np.sqrt(2)
    if sparse:
        assert isinstance(mat, sp.csc_matrix)
        ii, jj = mat.nonzero()
        if len(ii) == 0:
            # got an empty matrix -- this can happen depending on the parameter values.
            raise ValueError
        triu_mask = jj >= ii
        flat_indices = ravel_multi_index_triu([ii[triu_mask], jj[triu_mask]], mat.shape)  # type: ignore
        data = np.array(mat[ii[triu_mask], jj[triu_mask]]).flatten()  # type: ignore
        vec_size = int(mat.shape[0] * (mat.shape[0] + 1) / 2)  # type: ignore
        return sp.csr_matrix(
            (data, ([0] * len(flat_indices), flat_indices)), (1, vec_size)
        )
    else:
        return np.array(mat[np.triu_indices(n=mat.shape[0])]).flatten()  # type: ignore


def get_labels(p: str, zi: str, zj: str, var_dict: Dict[str, int]) -> List[str]:
    """Generate labels for matrix/vector elements based on variable sizes."""
    labels: List[str] = []
    size_i = var_dict[zi]
    size_j = var_dict[zj]
    if zi == zj:
        key_pairs = itertools.combinations_with_replacement(range(size_i), 2)
    else:
        key_pairs = itertools.product(range(size_i), range(size_j))
    for i, j in key_pairs:
        label = f"{p}-"
        label += f"{zi}:{i}." if size_i > 1 else f"{zi}."
        label += f"{zj}:{j}" if size_j > 1 else f"{zj}"
        labels.append(label)
    return labels
