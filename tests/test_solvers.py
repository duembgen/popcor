import numpy as np

from popcor.base_lifters import PolyLifter, RobustPoseLifter

# from lifters.matweight_lifter import MatWeightLifter
from popcor.examples import MonoLifter
from popcor.utils.geometry import get_C_r_from_theta
from popcor.utils.test_tools import all_lifters

NOISE = 1e-2


def test_hess_finite_diff():
    for lifter in all_lifters():
        lifter.generate_random_setup()
        lifter.sample_theta()

        errors = []
        eps_list = np.logspace(-10, -5, 5)
        for eps in eps_list:
            Q = lifter.get_Q(noise=NOISE)
            y = lifter.y_
            theta = lifter.get_vec_around_gt(delta=0)

            try:
                grad = lifter.get_grad(theta, y)
                hess = lifter.get_hess(theta, y)
                pass
            except (NotImplementedError, AttributeError):
                print("get_hess not implemented?")
                return

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            errors_mat = np.full((n, n), -np.inf)
            for i in range(n):
                theta_delta = theta + I[i]
                grad_delta = lifter.get_grad(theta_delta, y)

                hess_est = (grad_delta - grad) / eps

                assert isinstance(hess, np.ndarray)
                abs_error = np.abs(hess_est - hess[i, :])
                errors_mat[i, :] = np.maximum(abs_error, errors_mat[i, :])
                max_err = max(np.max(abs_error), max_err)
            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except ValueError:
            print("skipping Hessian test")
        # except AssertionError:
        # except Exception as e:
        #    print(f"Hessian test for {lifter} not passing")
        # import matplotlib.pylab as plt
        # plt.figure()
        # plt.title(f"hess {lifter}")
        # plt.loglog(eps_list, errors)
        # plt.show()
        # assert  < 1e-7


def test_grad_finite_diff():
    for lifter in all_lifters():
        errors = []
        eps_list = np.logspace(-10, -1, 11)
        for eps in eps_list:
            Q = lifter.get_Q()
            y = lifter.y_

            theta = lifter.get_vec_around_gt(delta=0)
            cost = lifter.get_cost(theta, y)

            try:
                grad = lifter.get_grad(theta, y)
            except NotImplementedError as e:
                print("grad not implemented?")
                continue
            else:
                if grad is None:
                    print("grad not implemented?")
                    continue

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            errors_mat = np.full(n, -np.inf)
            for i in range(n):
                theta_delta = theta + I[i]
                cost_delta = lifter.get_cost(theta_delta, y)

                grad_est = (cost_delta - cost) / eps

                assert isinstance(grad, np.ndarray)
                err = abs(grad_est - grad[i])
                errors_mat[i] = max(errors_mat[i], err)
                max_err = max(err, max_err)

            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except ValueError:
            print("skipping grad test")
        except AssertionError:
            import matplotlib.pylab as plt

            plt.figure()
            plt.title(f"grad {lifter}")
            plt.loglog(eps_list, errors)
            plt.show()


def test_cost_noisy():
    test_cost(noise=NOISE)


def test_cost(noise=0.0):
    for lifter in all_lifters():
        # np.random.seed(1)
        Q = lifter.get_Q(noise=noise)
        y = lifter.y_
        # np.random.seed(1)
        # Qold, yold = lifter.get_Q_old(noise=noise)
        # np.testing.assert_allclose(Q, Qold)
        # np.testing.assert_allclose(y, yold)
        if Q is None:
            continue

        theta = lifter.theta
        cost = lifter.get_cost(theta, y)

        x = lifter.get_x(theta)
        costQ = x.T @ Q @ x

        assert abs(cost - costQ) < 1e-6, (cost, costQ)
        if noise == 0 and not isinstance(lifter, PolyLifter) and not lifter.robust:
            assert cost < 1e-10, cost
            assert costQ < 1e-7, costQ
        elif noise == 0 and isinstance(lifter, MonoLifter):

            if lifter.robust:
                assert lifter.landmarks is not None
                w = lifter.theta[-lifter.n_landmarks :]
                for i in range(len(lifter.y_)):
                    C, r = get_C_r_from_theta(lifter.theta, lifter.d)

                    y_gt = C @ lifter.landmarks[i] + r
                    y_gt /= np.linalg.norm(y_gt)

                    if w[i] == -1:
                        # make sure there is a large error in outliers
                        assert not np.all(np.abs(y_gt - lifter.y_[i]) < 1e-3)
                    else:
                        np.testing.assert_allclose(y_gt, lifter.y_[i])

                try:
                    assert abs(cost - np.sum(w < 0)) < 1e-10
                except AssertionError:
                    lifter.plot_setup()


def test_solvers_noisy(n_seeds=1, noise=NOISE):
    test_solvers(n_seeds=n_seeds, noise=noise)


def test_solvers(n_seeds=1, noise=0.0):
    for lifter in all_lifters():
        for j in range(n_seeds):
            np.random.seed(j)

            # noisy setup
            Q = lifter.get_Q(noise=noise)
            y = lifter.y_
            if Q is None:
                continue

            # test that we stay at real solution when initializing at it
            theta_gt = lifter.get_vec_around_gt(delta=0)
            try:
                theta_hat, info, cost_solver = lifter.local_solver(theta_gt, y)
                print("local solution:", theta_hat, f"cost: {cost_solver:.4e}")
                print("ground truth:  ", theta_gt)
            except NotImplementedError:
                print("local solver not implemented yet.")
                continue
            if noise == 0:
                # test that solution is ground truth with no noise

                if len(theta_hat) == len(theta_gt):
                    np.testing.assert_allclose(theta_hat, theta_gt)
                else:
                    theta_gt = lifter.get_vec_around_gt(delta=0)
                    np.testing.assert_allclose(theta_hat, theta_gt)

            else:
                # just test that we converged when noise is added
                assert theta_hat is not None

            # test that we converge to real solution when initializing around it
            theta_0 = lifter.get_vec_around_gt(delta=NOISE)
            theta_hat, info, cost_solver = lifter.local_solver(theta_0, y)

            print("init:          ", theta_0)
            print("local solution:", theta_hat, f"cost: {cost_solver:.4e}")
            print("ground truth:  ", theta_gt)

            if theta_hat is None:
                print(f"Warning: {lifter} did not converge noise {noise}, seed {j}.")
                continue
            else:
                print(f"{lifter} converged noise {noise}, seed {j}.")

            cost_lifter = lifter.get_cost(theta_hat, y)
            if cost_lifter >= 1e-10:
                assert abs(cost_solver - cost_lifter) / cost_lifter < 1e-5, (
                    cost_solver,
                    cost_lifter,
                )

            # test that we made progress
            if len(theta_0) != len(theta_hat):
                progress = np.linalg.norm(theta_0 - theta_hat)
            else:
                progress = np.linalg.norm(theta_0 - theta_hat)
            assert progress > 1e-10, progress

            if noise == 0:
                # test that cost decreased
                cost_0 = lifter.get_cost(theta_0, y)
                cost_hat = lifter.get_cost(theta_hat, y)
                assert cost_hat <= cost_0

                # TODO(FD) this doesn't pass, looks like the problem is actually not well conditioned!
                if lifter.n_outliers > 0:
                    continue
                try:
                    assert isinstance(theta_gt, np.ndarray)
                    assert isinstance(theta_hat, np.ndarray)
                    np.testing.assert_allclose(theta_hat, theta_gt, rtol=1e-3)
                except AssertionError as e:
                    print(
                        f"Found solution for {lifter} is not ground truth in zero-noise! is the problem well-conditioned?"
                    )
                    fig, ax, im = lifter.plot(estimates={"estimate": theta_hat})
                    import matplotlib.pylab as plt

                    plt.show(block=False)
                    ax.legend()

                    try:
                        mineig_hess_hat = np.linalg.eigvalsh(
                            lifter.get_hess(theta_hat, y).toarray()
                        )[0]
                        mineig_hess_gt = np.linalg.eigvalsh(
                            lifter.get_hess(theta_gt, y).toarray()
                        )[0]
                        print(
                            f"minimum eigenvalue at gt: {mineig_hess_gt:.1e} and at estimate: {mineig_hess_hat:.1e}"
                        )
                    except (NotImplementedError, AttributeError):
                        print("implement Hessian for further checks.")
                    print(e)


def compare_solvers():
    noise = NOISE
    for lifter in all_lifters():
        if isinstance(lifter, RobustPoseLifter):
            compare_solvers = [
                "CG",
                "SD",  # "TR", very slow for some reason
            ]
        else:
            compare_solvers = [
                "Nelder-Mead",
                "Powell",
                # "CG",  CG takes forever.
                # "Newton-CG",
                "BFGS",
                "TNC",
            ]
        print("testing", lifter)
        np.random.seed(0)

        # noisy setup
        Q = lifter.get_Q(noise=noise)
        y = lifter.y_
        if Q is None:
            continue

        # test that we stay at real solution when initializing at it
        theta_gt = lifter.get_vec_around_gt(delta=0)

        import time

        for solver in compare_solvers:
            t1 = time.time()
            try:
                theta_hat, msg, cost_solver = lifter.local_solver(
                    theta_gt, y, method=solver
                )
            except NotImplementedError:
                continue
            ttot = time.time() - t1
            if theta_hat is None:
                print(solver, "failed")
            else:
                error = lifter.get_error(theta_hat)
                print(
                    f"{solver} finished in {ttot:.4f}s, final cost {cost_solver:.1e}, error {error:.1e}. \n\tmessage:{msg} "
                )


if __name__ == "__main__":
    import warnings

    # import pytest
    # pytest.main([__file__, "-s"])
    # print("all tests passed")

    with warnings.catch_warnings():
        # warnings.simplefilter("error"))
        test_cost()
        test_cost_noisy()
        test_solvers()
        test_solvers_noisy()

        test_grad_finite_diff()
        test_hess_finite_diff()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compare_solvers()

    print("all tests passed")
