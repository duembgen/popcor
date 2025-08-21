import numpy as np
import pytest

from popcor.examples import RotationLifter


@pytest.mark.parametrize("d", [2, 3])
def test_rank1_vs_rankd(d):
    """
    Make sure that the rank-1 and rank-d lifters give the same cost
    for the same measurements
    """
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
    cost2 = np.trace(x2.T @ Q2 @ x2)  # + (n_rot + n_abs + 1) * d
    assert abs(cost1 - cost2) < 1e-10

    # actual cost should be
    # 2 sum_ij tr(I) - 2 sum_ij tr(Ri.T@R_ij@R_j)
    #                ==========our cost =========
    cost_optimal = cost1 + 2 * (n_abs * n_rot + n_rot - 1) * d
    assert abs(cost_optimal) < 1e-8


@pytest.mark.parametrize("d", [2, 3])
def test_measurements(d, level="no"):
    """
    Make sure that the forward model for measurements is correct
    """
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


if __name__ == "__main__":
    test_measurements(d=2, level="no")
    test_measurements(d=2, level="bm")
    test_measurements(d=3, level="no")
    test_measurements(d=3, level="bm")
    test_rank1_vs_rankd(d=2)
    test_rank1_vs_rankd(d=3)
    print("all tests passed")
