"""
Tests for RotationLifter behaviors including measurement generation, local solvers,
and semidefinite programming (SDP) relaxations. Covers consistency between rank-1
and rank-d lifters, correctness of the forward measurement model, convergence of
a local solver under varying noise levels, and recovery via an SDP relaxation.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp

from popcor.examples import RotationLifter
from popcor.utils.plotting_tools import plot_matrix

PLOT = False


def plot_matrices(A_known, Q):
    n_cols = min(1 + len(A_known), 10)
    n_rows = math.ceil((len(A_known) + 1) / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3 * n_cols, 3 * n_rows)
    axs = axs.flatten()
    for i in range(len(A_known)):
        assert isinstance(A_known[i], sp.csc_matrix)
        plot_matrix(A_known[i].toarray(), ax=axs[i], title=f"A{i} ", colorbar=False)
    fig = plot_matrix(Q.toarray(), ax=axs[i + 1], title="Q", colorbar=False)
    [axs[j].axis("off") for j in range(i + 1, len(axs))]


@pytest.mark.parametrize("d", [2, 3])
def test_rank1_vs_rankd(d):
    """Ensure rank-1 and rank-d RotationLifter formulations yield identical costs."""
    n_rot = 3
    n_abs = 2

    np.random.seed(0)
    r1 = RotationLifter(level="no", n_abs=n_abs, n_rot=n_rot, d=d)

    np.random.seed(0)
    r2 = RotationLifter(level="bm", n_abs=n_abs, n_rot=n_rot, d=d)

    np.testing.assert_array_equal(r1.theta, r2.theta)

    y = r1.simulate_y(noise=1e-5)

    x1 = r1.get_x()
    x2 = r2.get_x()

    Q1 = r1.get_Q_from_y(y, output_poly=False)
    Q2 = r2.get_Q_from_y(y, output_poly=False)

    cost1 = x1.T @ Q1 @ x1
    cost2 = np.trace(x2.T @ Q2 @ x2)
    assert abs(cost1 - cost2) < 1e-10

    # actual optimal cost should be
    # 2 sum_ij tr(I) - 2 sum_ij tr(Ri.T@R_ij@R_j)
    cost_optimal = cost1 + 2 * (n_abs * n_rot + n_rot - 1) * d
    assert abs(cost_optimal) < 1e-8


@pytest.mark.parametrize("d", [2, 3])
def test_measurements(d, level="no"):
    """Verify that the simulated measurements match the lifter's internal rotations."""
    np.random.seed(0)
    n_abs = 2
    n_rot = 3
    d = 3
    lifter = RotationLifter(
        d=d, n_abs=n_abs, n_rot=n_rot, sparsity="chain", level=level
    )
    y = lifter.simulate_y(noise=1e-10)

    for key, R in y.items():
        if isinstance(key, int):
            # unary factor
            for Rk in R:
                R_i = lifter.theta[:, key * lifter.d : (key + 1) * lifter.d]
                np.testing.assert_allclose(R_i, Rk, atol=1e-5)
        elif isinstance(key, tuple):
            # binary factor
            i, j = key
            R_i = lifter.theta[:, i * lifter.d : (i + 1) * lifter.d]
            R_j = lifter.theta[:, j * lifter.d : (j + 1) * lifter.d]
            np.testing.assert_allclose(R_i.T @ R_j, R, atol=1e-5)

    x = lifter.get_x()
    theta = lifter.get_theta(x)
    np.testing.assert_allclose(theta, lifter.theta)


def test_solve_local():
    """Test the local solver's convergence from different initializations and noise levels."""
    d = 3
    n_abs = 1
    n_rot = 4
    for noise in [0.0, 0.2]:
        np.random.seed(2)
        lifter = RotationLifter(
            d=d, n_abs=n_abs, n_rot=n_rot, sparsity="chain", level="no"
        )

        y = lifter.simulate_y(noise=noise)

        theta_gt, *_ = lifter.local_solver(lifter.theta, y, verbose=False)
        estimates = {"init gt": theta_gt}
        for i in range(10):
            theta_init = lifter.sample_theta()
            theta_i, *_ = lifter.local_solver(theta_init, y, verbose=False)

            # make sure we are starting relatively far from ground truth
            assert np.any(np.abs(theta_init - theta_gt) > 1e-1)
            estimates[f"init random {i}"] = theta_i

            # check convergence tolerance depending on noise
            if noise == 0.0:
                np.testing.assert_allclose(theta_i, lifter.theta, atol=1e-5)
            else:
                # 0.5 is found empirically
                np.testing.assert_allclose(theta_i, lifter.theta, atol=0.5)

        if PLOT:
            _, ax = lifter.plot(estimates=estimates)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles[:3], labels[:3], loc="lower left", bbox_to_anchor=(0.0, 1.0)
            )
            plt.show(block=False)
    print("done")


@pytest.mark.parametrize(
    "example_type, expect_equal_minima",
    [
        ("A", True),
        ("B", False),
    ],
)
def test_so2_example_minima_structure(example_type, expect_equal_minima):
    """Validate deterministic SO(2) tutorial examples have intended minima patterns."""
    lifter = RotationLifter.create_example(example_type=example_type)

    angles = np.linspace(0.0, 2.0 * np.pi, 4001)
    costs = np.array([lifter.get_cost(lifter.so2_theta(t)) for t in angles])

    minima_idx = []
    for i in range(1, len(angles) - 1):
        if costs[i] <= costs[i - 1] and costs[i] <= costs[i + 1]:
            if minima_idx and (i - minima_idx[-1]) < 20:
                if costs[i] < costs[minima_idx[-1]]:
                    minima_idx[-1] = i
            else:
                minima_idx.append(i)

    assert len(minima_idx) == 2
    c1, c2 = float(costs[minima_idx[0]]), float(costs[minima_idx[1]])

    if expect_equal_minima:
        assert abs(c1 - c2) < 1e-5
    else:
        assert abs(c1 - c2) > 1e-3


@pytest.mark.parametrize("example_type", ["A", "B"])
def test_so2_example_sdp_recovery(example_type):
    """Check SDP behavior for deterministic SO(2) examples.

    Type A is symmetric with two equal global minima, so the SDP optimum is generally
    not rank-1 in the angle-moment block. Type B has a unique global minimum and should
    be near rank-1.
    """
    lifter = RotationLifter.create_example(example_type=example_type)

    Q = lifter.get_Q()
    A_known, b_known = lifter.get_A_known(add_redundant=True)
    A_0, b_0 = lifter.get_A0()
    constraints = list(zip(A_0 + A_known, b_0 + b_known))

    X, info_sdp = solve_sdp(Q, constraints, verbose=False)
    _, info_rank = rank_project(X, p=1)

    angles = np.linspace(-np.pi, np.pi, 5001)
    costs = np.array([lifter.get_cost(lifter.so2_theta(t)) for t in angles])
    c_min = float(np.min(costs))

    # SDP lower bound should match the global minimum cost very tightly.
    assert abs(float(info_sdp["cost"]) - c_min) < 1e-6

    if example_type == "A":
        # Symmetric two-minimum case should not force rank-1 recovery.
        assert info_rank["EVR"] < 10.0
    else:
        # Unique minimum case should be near rank-1.
        assert info_rank["EVR"] > 1e6


@pytest.mark.parametrize("level", ["no", "bm"])
def test_solve_sdp(level):
    """Solve the SDP relaxation and compare the recovered rotations to ground truth."""
    d = 3
    n_abs = 1
    n_rot = 5
    estimates = {}
    for noise in [0, 1e-3, 0.2]:
        np.random.seed(2)
        lifter = RotationLifter(
            d=d, n_abs=n_abs, n_rot=n_rot, sparsity="chain", level=level
        )

        y = lifter.simulate_y(noise=noise)
        Q = lifter.get_Q_from_y(y=y, output_poly=False)

        assert isinstance(Q, sp.csc_matrix)
        A_known, b_known = lifter.get_A_known()
        A_0, b_0 = lifter.get_A0()
        constraints = list(zip(A_0 + A_known, b_0 + b_known))

        if noise == 0 and PLOT:
            plot_matrices(A_known, Q)

        X, info = solve_sdp(Q, constraints, verbose=False)

        if lifter.level == "no":
            x, info_rank = rank_project(X, p=1)
            theta_sdp = lifter.get_theta(x)
        else:
            X, info_rank = rank_project(X, p=lifter.d)
            theta_sdp = lifter.get_theta(X)

        print(f"EVR at noise {noise:.2f}: {info_rank['EVR']:.2e}")
        if noise <= 1:
            assert info_rank["EVR"] > 1e8

        error = lifter.get_error(theta_sdp)
        if noise == 0.0:
            np.testing.assert_allclose(theta_sdp, lifter.theta, atol=1e-8)
            assert error < 1e-10
        elif noise < 1e-2:
            assert error < 1e-2

        estimates.update({"init gt": lifter.theta, f"SDP noise {noise:.2f}": theta_sdp})

    if PLOT:
        fig, ax = lifter.plot(estimates=estimates)
        _, ax = lifter.plot(estimates=estimates)
        plt.show(block=False)
    return


if __name__ == "__main__":
    test_solve_sdp(level="no")
    test_solve_sdp(level="bm")
    test_solve_local()

    test_measurements(d=2, level="no")
    test_measurements(d=2, level="bm")
    test_measurements(d=3, level="no")
    test_measurements(d=3, level="bm")

    test_rank1_vs_rankd(d=2)
    test_rank1_vs_rankd(d=3)

    print("all tests passed")
