import numpy as np

from popr import AutoTemplate
from popr.examples import RangeOnlyLocLifter, Stereo1DLifter

# random seed, for reproducibility
SEED = 3


def test_stereo_1d():
    np.random.seed(SEED)
    lifter = Stereo1DLifter(n_landmarks=5, param_level="p")
    learner = AutoTemplate(
        lifter=lifter,
    )
    data, success = learner.run(
        use_known=False,
        use_incremental=False,
        variable_list=[["h", "x"], ["h", "x", "z_0"], ["h", "x", "z_0", "z_1"]],
    )
    try:
        assert success
    except AssertionError:
        learner.save_matrices_poly()


def test_range_only():
    np.random.seed(SEED)
    for level in ["no", "quad"]:
        lifter = RangeOnlyLocLifter(n_positions=4, n_landmarks=5, d=2, level=level)
        learner = AutoTemplate(
            lifter=lifter,
        )
        data, success = learner.run(
            use_known=False,
            use_incremental=False,
            variable_list=[["h", "x_0"], ["h", "x_0", "z_0"]],
        )
        assert success

        learner = AutoTemplate(
            lifter=lifter,
        )
        data, success = learner.run(
            use_known=True,
            use_incremental=False,
            variable_list=[["h", "x_0"], ["h", "x_0", "z_0"]],
        )
        assert success


if __name__ == "__main__":
    test_stereo_1d()
    test_range_only()
    print("all tests passed")
