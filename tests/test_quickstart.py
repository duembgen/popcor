"""Module containing all tests featured in the documentation."""

import numpy as np
import pytest


def test_setup_problem():
    """Test the problem setup example from the documentation."""
    from popcor.examples import Poly4Lifter

    lifter = Poly4Lifter()

    Q = lifter.get_Q()

    # theta corresponds to the ground truth; in this case, the global minimum.
    x = lifter.get_x(lifter.theta).flatten()
    cost_optimum = float(x.T @ Q @ x)

    # the cost at any other randomly sampled point has to be larger.
    for i in range(10):
        theta_random = lifter.sample_theta()
        x_random = lifter.get_x(theta_random).flatten()
        assert float(x_random.T @ Q @ x_random) > cost_optimum


def test_solve_sdp():
    """Test the SDP example from the documentation."""
    from cert_tools.linalg_tools import rank_project
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.examples import Poly4Lifter

    lifter = Poly4Lifter()

    # the cost matrix
    Q = lifter.get_Q()

    # the equality constraints
    A_known, b_known = lifter.get_A_known()

    # the homogenization constraint
    A_0, b_0 = lifter.get_A0()
    constraints = list(zip(A_0 + A_known, b_0 + b_known))

    X, info = solve_sdp(Q, constraints)
    assert X is not None

    # if X is rank one, the global optimum can be found in element X_10 of the matrix.
    theta_pick = X[1, 0]
    assert abs(theta_pick - lifter.theta) < 1e-5

    # We can also first extract the rank-1 estimate (X=xx') and then extract theta.
    x, info_rank = rank_project(X, p=1)
    theta_round = x[1]

    assert abs(theta_round - lifter.theta) < 1e-5


def test_autotight():
    """Test the AutoTight example from the documentation."""
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.auto_tight import AutoTight
    from popcor.examples.stereo1d_lifter import Stereo1DLifter

    lifter = Stereo1DLifter(n_landmarks=5)

    auto_tight = AutoTight()

    # solve the SDP -- it is not cost tight!
    Q = lifter.get_Q(noise=1e-1)
    A_known, b_known = lifter.get_A_known()
    A_0, b_0 = lifter.get_A0()

    # solve locally starting at ground truth
    x, info_local, cost = lifter.local_solver(lifter.theta, y=lifter.y_)
    assert x is not None
    assert info_local["success"]

    constraints = list(zip(A_0 + A_known, b_0 + b_known))
    X, info_sdp = solve_sdp(Q, constraints)

    gap = auto_tight.get_duality_gap(
        info_local["cost"], info_sdp["cost"], relative=True
    )
    assert gap > 0.1

    # learn matrices and solve again
    A_learned, b_learned = auto_tight.get_A_learned(lifter)
    A_0, b_0 = lifter.get_A0()
    constraints = list(zip(A_0 + A_learned, b_0 + b_learned))
    X, info_sdp_learned = solve_sdp(Q, constraints)
    gap = auto_tight.get_duality_gap(
        info_local["cost"], info_sdp_learned["cost"], relative=True
    )

    # note that the gap can be slightly negative because of mismatch in convergence tolerances etc.
    assert abs(gap) < 1e-2

    # problem: if we change landmarks, the constraints do not generalize!
    new_lifter = Stereo1DLifter(n_landmarks=5)
    assert not np.all(new_lifter.landmarks == lifter.landmarks)

    # assert that the following line raises an error
    # this is exactly why we need to use auto_template here!
    with pytest.raises(Exception):
        new_lifter.test_constraints(A_learned, errors="raise", n_seeds=1)


def test_autotemplate():
    """Test the AutoTemplate example from the documentation."""
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.auto_template import AutoTemplate
    from popcor.auto_tight import AutoTight
    from popcor.examples.stereo1d_lifter import Stereo1DLifter

    # important: we need to use param_level="p", otherwise the parameters
    # are not factored out and the constraints are not generalizable.
    lifter = Stereo1DLifter(n_landmarks=3, param_level="p")

    # learn the template matrices
    auto_template = AutoTemplate(lifter)
    data, success = auto_template.run(use_known=False, plot=True)
    assert success

    # apply the templates to a different lifter
    new_lifter = Stereo1DLifter(n_landmarks=5, param_level="p")
    constraints = auto_template.apply(new_lifter, use_known=True)
    Q = new_lifter.get_Q(noise=1e-1)

    # adds homogenization constraint and all b_i terms
    X, info_sdp = solve_sdp(Q, constraints)

    # evaluate duality gap
    x, info_local, cost = new_lifter.local_solver(new_lifter.theta, y=new_lifter.y_)
    gap = AutoTight.get_duality_gap(info_local["cost"], info_sdp["cost"], relative=True)
    # note that the gap can be slightly negative because of mismatch in convergence tolerances etc.
    assert abs(gap) < 1e-2


def test_autotemplate_literal():
    """Here, we used AutoTemplate to interpret the templates (i.e., implemented get_A_known_redundant)"""
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.auto_tight import AutoTight
    from popcor.examples.stereo1d_lifter import Stereo1DLifter

    new_lifter = Stereo1DLifter(n_landmarks=5, param_level="p")

    Q = new_lifter.get_Q(noise=1e-1)
    A_0, b_0 = new_lifter.get_A0()
    A_known, b_known = new_lifter.get_A_known()
    A_red, b_red = new_lifter.get_A_known_redundant()
    constraints = list(zip(A_0 + A_known + A_red, b_0 + b_known + b_red))

    X, info_sdp = solve_sdp(Q, constraints)

    # evaluate duality gap
    x, info_local, cost = new_lifter.local_solver(new_lifter.theta, y=new_lifter.y_)
    gap = AutoTight.get_duality_gap(info_local["cost"], info_sdp["cost"], relative=True)
    # note that the gap can be slightly negative because of mismatch in convergence tolerances etc.
    assert abs(gap) < 1e-2


if __name__ == "__main__":
    import warnings

    # make sure warnings raise errors, for debugging
    warnings.simplefilter("error")
    warnings.simplefilter("ignore", UserWarning)

    test_setup_problem()
    test_solve_sdp()
    test_autotight()
    test_autotemplate()
    test_autotemplate_literal()
    print("all tests passed")
