import numpy as np

from popcor.examples import RotationLifter


def test_rank1_vs_rankd():
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


if __name__ == "__main__":

    test_rank1_vs_rankd()
