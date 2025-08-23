import warnings

import numpy as np

from popcor import AutoTight
from popcor.examples import Stereo1DLifter, Stereo2DLifter, Stereo3DLifter
from popcor.utils.test_tools import (
    all_lifters,
    constraints_test_with_tol,
    example_lifters,
)

# random seed, for reproducibility
SEED = 3


def test_constraints():
    # testing qrp, our go-to method, for all example lifters
    for lifter in all_lifters(SEED):
        A_learned, b_learned = AutoTight.get_A_learned(lifter=lifter, method="qrp")
        constraints_test_with_tol(lifter, A_learned, b_learned, tol=1e-4)


def test_constraints_params():
    # for a couple of representative lifters, checking that all parameter levels are working
    for param_level in ["no", "p", "ppT"]:
        for lifter in example_lifters(seed=SEED, param_level=param_level):
            A_learned, b_learned = AutoTight.get_A_learned(lifter=lifter, verbose=False)
            constraints_test_with_tol(lifter, A_learned, b_learned, tol=1e-4)


def test_constraints_methods():
    # for a couple of representative lifters, checking that all nullspace methods work.
    for lifter in example_lifters(seed=SEED, param_level="p"):
        num_learned = None
        for method in ["qrp", "svd", "qr"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                A_learned, b_learned = AutoTight.get_A_learned(
                    lifter=lifter, verbose=False, method=method
                )
            constraints_test_with_tol(lifter, A_learned, b_learned, tol=1e-4)

            # make sure each method finds the same number of matrices
            if num_learned is None:
                num_learned = len(A_learned)
            else:
                assert len(A_learned) == num_learned


def test_constraints_stereo():
    np.random.seed(0)
    warnings.filterwarnings("ignore", category=UserWarning)
    for param_level in ["p", "ppT"]:
        print(f"learning Stereo1D, {param_level}")
        lifter = Stereo1DLifter(n_landmarks=1, param_level=param_level)
        A_learned, b_learned = AutoTight.get_A_learned(lifter)
        lifter.test_constraints(A_learned, errors="raise", n_seeds=1)

        print(f"learning Stereo2D, {param_level}")
        lifter = Stereo2DLifter(n_landmarks=1, param_level=param_level, level="urT")
        A_learned, b_learned = AutoTight.get_A_learned(lifter)
        lifter.test_constraints(A_learned, errors="raise", n_seeds=1)

        print(f"learning Stereo3D, {param_level}")
        lifter = Stereo3DLifter(n_landmarks=1, param_level=param_level, level="urT")
        A_learned, b_learned = AutoTight.get_A_learned(lifter)
        lifter.test_constraints(A_learned, errors="raise", n_seeds=1)


if __name__ == "__main__":
    test_constraints_stereo()
    test_constraints()
    test_constraints_params()
    test_constraints_methods()
    print("all tests passed")
