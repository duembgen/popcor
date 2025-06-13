import numpy as np

from popcor.examples import RotationLifter


def test_rank1_vs_rankd():
    """make sure that the rank-1 and rank-d lifters give the same cost
    for the same measurements"""
    np.random.seed(0)
    r1 = RotationLifter(level="no", n_meas=0, n_rot=3, d=2)

    np.random.seed(0)
    r2 = RotationLifter(level="bm", n_meas=0, n_rot=3, d=2)

    np.testing.assert_array_equal(r1.theta, r2.theta)

    y = r1.simulate_y(noise=1e-5)

    x1 = r1.get_x()
    x2 = r2.get_x()

    Q1 = r1.get_Q_from_y(y)
    Q2 = r2.get_Q_from_y(y)

    cost1 = x1.T @ Q1 @ x1
    cost2 = np.trace(x2.T @ Q2 @ x2)
    assert abs(cost1 - cost2) < 1e-10

    # actual cost should be
    # 2 sum_ij tr(I) - 2 sum_ij tr(Ri.T@R_ij@R_j)
    #                ==========our cost =========
    cost_optimal = cost1 + 2 * len(y) * r1.d
    assert abs(cost_optimal) < 1e-8


def test_measurements(level="no"):
    """make sure that the forward model for measurements is correct"""
    lifter = RotationLifter(d=2, n_meas=0, n_rot=3, sparsity="chain", level=level)
    y = lifter.simulate_y(noise=1e-10)

    for (i, j), R in y.items():
        R_i = lifter.theta[i * lifter.d : (i + 1) * lifter.d, :]
        R_j = lifter.theta[j * lifter.d : (j + 1) * lifter.d, :]
        np.testing.assert_allclose(R_i @ R_j.T, R, atol=1e-5)

    x = lifter.get_x()
    rank = x.shape[1] if np.ndim(x) == 2 else 1

    theta = lifter.get_theta(x)
    for (i, j), R in y.items():
        R_i = theta[i * lifter.d : (i + 1) * lifter.d, :]
        R_j = theta[j * lifter.d : (j + 1) * lifter.d, :]
        np.testing.assert_allclose(R_i @ R_j.T, R, atol=1e-5)


if __name__ == "__main__":
    test_measurements(level="no")
    test_measurements(level="bm")
    test_rank1_vs_rankd()
    print("all tests passed")
