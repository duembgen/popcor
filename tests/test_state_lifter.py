import numpy as np
import pytest
import scipy.sparse as sp

from popcor.base_lifters import StateLifter, StereoLifter
from popcor.examples import Stereo2DLifter, Stereo3DLifter
from popcor.utils.common import (
    get_vec,
    ravel_multi_index_triu,
    unravel_multi_index_triu,
)
from popcor.utils.test_tools import all_lifters, constraints_test_with_tol


def pytest_configure():
    # global variables
    pytest.A_learned = {}  # type: ignore
    for lifter in all_lifters():
        pytest.A_learned[str(lifter)] = None  # type: ignore


def test_ravel():
    shape = (5, 5)
    # test diagonal elements
    for i in range(shape[0]):
        idx = np.array([i])
        flat_idx = ravel_multi_index_triu([idx, idx], shape=shape)
        i_test, j_test = unravel_multi_index_triu(flat_idx, shape=shape)

        assert idx == i_test[0]
        assert idx == j_test[0]

    # test random elements
    for seed in range(100):
        np.random.seed(seed)
        i = np.random.randint(low=0, high=shape[0] - 1, size=1)
        j = np.random.randint(low=i, high=shape[0] - 1, size=1)

        flat_idx = ravel_multi_index_triu([i, j], shape=shape)
        i_test, j_test = unravel_multi_index_triu(flat_idx, shape=shape)

        assert i == i_test[0]
        assert j == j_test[0]


def test_known_constraints():
    raise_error = False
    for lifter in all_lifters():
        try:
            A_known, b_known = lifter.get_A_known()
        except ValueError:
            raise_error = True
            print(f"Error in {lifter}")
            continue

        constraints_test_with_tol(lifter, A_known, b_known, tol=1e-10)

        B_known = lifter.get_B_known()
        x = lifter.get_x(theta=lifter.theta)
        for Bi in B_known:
            assert x.T @ Bi @ x <= 0
    if raise_error:
        raise ValueError("Some lifters could not provide known constraints.")


def test_constraint_rank():
    raise_error = False
    for lifter in all_lifters():
        assert isinstance(lifter, StateLifter)
        try:
            A_known, b_known = lifter.get_A_known(add_redundant=False)
            pass
        except TypeError:
            print(f"lifter {lifter} does not support add_redundant!")
            A_known, b_known = lifter.get_A_known()
            raise_error = True

        rank = lifter.get_constraint_rank(A_known, b_known=b_known)
        assert rank == len(A_known)
    if raise_error:
        raise ValueError("Some lifters could not provide known constraints.")


def test_vec_mat():
    """Make sure that we can go back and forth from vec to mat."""
    for lifter in all_lifters():
        try:
            A_known, b_known = lifter.get_A_known()
        except AttributeError:
            print(f"could not get A_known of {lifter}")
            A_known = []

        for A in A_known:
            a_dense = get_vec(A.toarray())
            a_sparse = get_vec(A)
            assert isinstance(a_dense, np.ndarray)
            assert isinstance(a_sparse, np.ndarray)
            np.testing.assert_allclose(a_dense, a_sparse)

            # get_vec multiplies off-diagonal elements by sqrt(2)
            a = get_vec(A)

            A_test = lifter.get_mat(a, sparse=False)
            assert isinstance(A_test, np.ndarray)
            np.testing.assert_allclose(A.toarray(), A_test)

            # get_mat divides off-diagonal elements by sqrt(2)
            A_test = lifter.get_mat(a, sparse=True)
            assert isinstance(A_test, sp.csr_array)
            np.testing.assert_allclose(A.toarray(), A_test.toarray())

            a_poly = lifter.convert_a_to_polyrow(a)
            a_test = lifter.convert_polyrow_to_a(a_poly)
            np.testing.assert_allclose(np.asarray(a), np.asarray(a_test))


def test_levels():
    for level in StereoLifter.LEVELS:
        lifter_2d = Stereo2DLifter(n_landmarks=3, level=level)

        # inside below function we tests that dimensions are consistent.
        lifter_2d.get_x()

        lifter_3d = Stereo3DLifter(n_landmarks=3, level=level)
        lifter_3d.get_x()


pytest_configure()

if __name__ == "__main__":
    # pytest.main([__file__, "-s"])
    # print("all tests passed")

    test_known_constraints()

    test_constraint_rank()
    test_ravel()
    test_vec_mat()
