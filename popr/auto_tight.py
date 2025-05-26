import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from cert_tools.linalg_tools import find_dependent_columns, get_nullspace

from popr.utils.common import get_vec
from popr.utils.constraint import Constraint
from popr.utils.plotting_tools import (
    add_colorbar,
    initialize_discrete_cbar,
    plot_singular_values,
)


class AutoTight(object):
    """Class for automatic constraint generation."""

    # consider singular value zero below this
    EPS_SVD = 1e-5

    # basis pursuit method, can be
    # - qr: qr decomposition
    # - qrp: qr decomposition with permutations (sparser), recommended
    # - svd: svd
    METHOD = "qrp"

    # normalize learned Ai or not
    NORMALIZE = False

    # how much to oversample (>= 1)
    FACTOR = 1.2

    # number of times we remove bad samples from data matrix
    N_CLEANING_STEPS = 1  # was 3

    # maximum number of iterations of local solver
    LOCAL_MAXITER = 100

    # find and remove linearly dependent constraints
    REDUCE_DEPENDENT = False

    def __init__(self):
        pass

    @staticmethod
    def clean_Y(basis_new, Y, s, plot=False):
        errors = np.abs(basis_new @ Y.T)  # Nb x n x n x Ns = Nb x Ns
        if np.all(errors < 1e-10):
            return []
        bad_bins = np.unique(np.argmax(errors, axis=1))
        if plot:
            fig, ax = plt.subplots()
            ax.semilogy(np.min(errors, axis=1))
            ax.semilogy(np.max(errors, axis=1))
            ax.semilogy(np.median(errors, axis=1))
            ax.semilogy(s)
        return bad_bins

    @staticmethod
    def test_S_cutoff(S, corank, eps_svd=None):
        if eps_svd is None:
            eps_svd = AutoTight.EPS_SVD
        if corank > 1:
            try:
                assert abs(S[-corank]) / eps_svd < 1e-1  # 1e-1  1e-10
                assert abs(S[-corank - 1]) / eps_svd > 10  # 1e-11 1e-10
            except AssertionError:
                print(f"there might be a problem with the chosen threshold {eps_svd}:")
                print(S[-corank], eps_svd, S[-corank - 1])

    @staticmethod
    def get_basis_sparse(
        lifter, var_list, param_list, A_known=[], plot=False, eps_svd=None
    ):

        Y = AutoTight.generate_Y_sparse(
            lifter, var_subset=var_list, param_subset=param_list, factor=1.0
        )
        basis, S = AutoTight.get_basis(lifter, Y, A_known=A_known, eps_svd=eps_svd)
        AutoTight.test_S_cutoff(S, corank=basis.shape[0], eps_svd=eps_svd)
        constraints = []
        for i, b in enumerate(basis):
            constraints.append(
                Constraint.init_from_b(
                    i,
                    b,
                    mat_var_dict=var_list,
                    mat_param_dict=param_list,
                    convert_to_polyrow=False,
                    known=False,
                )
            )
        if plot:
            plot_matrix = np.vstack([t.b_[None, :] for t in constraints])

            cmap, norm, colorbar_yticks = initialize_discrete_cbar(plot_matrix)
            X_dim = lifter.get_dim_X(var_list)
            fig, ax = plt.subplots()
            ax.axvline(X_dim - 0.5, color="red")
            im = ax.matshow(plot_matrix, cmap=cmap, norm=norm)
            ax.set_title(f"{var_list}, {param_list}")
            cax = add_colorbar(fig, ax, im)
            if colorbar_yticks is not None:
                cax.set_yticklabels(colorbar_yticks)
            plt.show(block=False)
        return constraints

    @staticmethod
    def get_A_learned(
        lifter,  #:StateLifter,
        A_known=[],
        var_dict=None,
        method=METHOD,
        verbose=False,
    ) -> list:
        """Generate list of learned constraints by sampling the lifter.

        :param lifter: StateLifter object
        :param A_known: list of known constraints, if given, will generate basis that is orthogonal to these given constraints.
        :param var_dict: variable dictionary, if None, will use all variables
        :param method: method to use for basis generation, can be 'qr', 'qrp', or 'svd'. 'qrp' is recommended.
        :param verbose: if True, will print timing information

        :return: list of learned constraints.
        """
        import time

        t1 = time.time()
        Y = AutoTight.generate_Y(lifter, var_subset=var_dict, factor=1.0)
        if verbose:
            print(f"generate Y ({Y.shape}): {time.time() - t1:4.4f}")
        t1 = time.time()
        basis, S = AutoTight.get_basis(lifter, Y, A_known=A_known, method=method)
        if verbose:
            print(f"get basis ({basis.shape})): {time.time() - t1:4.4f}")
        t1 = time.time()
        A_learned = AutoTight.generate_matrices(lifter, basis, var_dict=var_dict)
        if verbose:
            print(f"get matrices ({len(A_learned)}): {time.time() - t1:4.4f}")
        return A_learned

    @staticmethod
    def get_A_learned_simple(
        lifter,  #:StateLifter,
        A_known=[],
        var_dict=None,
        method=METHOD,
        verbose=False,
    ) -> list:
        """Simplified version of get_A_learned that does not consider parameters."""
        import time

        t1 = time.time()
        Y = AutoTight.generate_Y_simple(lifter, var_subset=var_dict, factor=1.5)
        if verbose:
            print(f"generate Y ({Y.shape}): {time.time() - t1:4.4f}")
        t1 = time.time()
        if len(A_known):
            basis_known = np.vstack(
                [
                    np.asarray(get_vec(Ai.get_matrix(var_dict)))
                    for Ai in A_known
                    if get_vec(Ai.get_matrix(var_dict)) is not None
                ]
            ).T
        else:
            basis_known = None
        basis, S = AutoTight.get_basis(
            lifter, Y, basis_known=basis_known, method=method
        )
        if verbose:
            print(f"get basis ({basis.shape})): {time.time() - t1:4.4f}")
        t1 = time.time()
        A_learned = AutoTight.generate_matrices_simple(lifter, basis, var_dict=var_dict)
        if verbose:
            print(f"get matrices ({len(A_learned)}): {time.time() - t1:4.4f}")
        return A_learned

    @staticmethod
    def generate_Y_simple(lifter, var_subset, factor):
        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_X(var_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            x = lifter.get_x(theta=theta, parameters=None, var_subset=var_subset)
            X = np.outer(x, x)
            Y[seed, :] = get_vec(X)
        return Y

    @staticmethod
    def generate_Y_sparse(lifter, var_subset, param_subset, factor=FACTOR, ax=None):
        from popr.base_lifters import StateLifter

        assert isinstance(lifter, StateLifter)
        assert lifter.HOM in param_subset

        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_Y(var_subset, param_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            parameters = lifter.sample_parameters(theta)

            if seed < 10 and ax is not None:
                if np.ndim(lifter.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = lifter.get_x(theta=theta, parameters=parameters, var_subset=var_subset)
            X = np.outer(x, x)

            # generates [1*x, a1*x, ..., aK*x]
            p = lifter.get_p(parameters=parameters, param_subset=param_subset)
            Y[seed, :] = np.kron(p, get_vec(X))
        return Y

    @staticmethod
    def generate_Y(lifter, factor=FACTOR, ax=None, var_subset=None, param_subset=None):
        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_Y(var_subset, param_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            parameters = lifter.sample_parameters(theta)
            if seed < 10 and ax is not None:
                if np.ndim(lifter.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = lifter.get_x(theta=theta, parameters=parameters, var_subset=var_subset)
            X = np.outer(x, x)

            # generates [1*x, a1*x, ..., aK*x]
            p = lifter.get_p(parameters=parameters, param_subset=param_subset)
            assert p[0] == 1
            Y[seed, :] = np.kron(p, get_vec(X))
        return Y

    @staticmethod
    def get_basis(
        lifter,
        Y,
        A_known: list = [],
        basis_known: np.ndarray | None = None,
        method=METHOD,
        eps_svd=None,
    ):
        """Generate basis from lifted state matrix Y.

        :param A_known: if given, will generate basis that is orthogonal to these given constraints.

        :return: basis, S
        """
        if eps_svd is None:
            eps_svd = AutoTight.EPS_SVD

        # if there is a known list of constraints, add them to the Y so that resulting nullspace is orthogonal to them
        if basis_known is not None:
            if len(A_known):
                print(
                    "Warning: ignoring given A_known because basis_all is also given."
                )
            Y = np.vstack([Y, basis_known.T])
        elif len(A_known):
            A = np.vstack(
                [lifter.augment_using_zero_padding(get_vec(a)) for a in A_known]
            )
            Y = np.vstack([Y, A])

        basis, info = get_nullspace(Y, method=method, tolerance=eps_svd)

        basis[np.abs(basis) < lifter.EPS_SPARSE] = 0.0
        return basis, info["values"]

    @staticmethod
    def generate_matrices_simple(
        lifter,
        basis,
        normalize=NORMALIZE,
        sparse=True,
        trunc_tol=1e-10,
        var_dict=None,
    ):
        """
        Generate constraint matrices from the rows of the nullspace basis matrix.
        """
        try:
            n_basis = len(basis)
        except Exception:
            n_basis = basis.shape[0]

        if isinstance(var_dict, list):
            var_dict = lifter.get_var_dict(var_dict)

        from popr.base_lifters import StateLifter

        assert isinstance(lifter, StateLifter)

        A_list = []
        for i in range(n_basis):
            ai = basis[i]
            Ai = lifter.get_mat(ai, sparse=sparse, correct=True, var_dict=None)
            # Normalize the matrix
            if normalize and not sparse:
                # Ai /= np.max(np.abs(Ai))
                assert isinstance(Ai, np.ndarray)
                Ai /= np.linalg.norm(Ai, p=2)  # type: ignore
            elif normalize and sparse:
                Ai /= splinalg.norm(Ai, ord="fro")
            # Sparsify and truncate
            if sparse:
                Ai.eliminate_zeros()  # type: ignore
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0  # type: ignore
            # add to list
            A_list.append(Ai)
        return A_list

    @staticmethod
    def generate_matrices(
        lifter,
        basis,
        normalize=NORMALIZE,
        sparse=True,
        trunc_tol=1e-10,
        var_dict=None,
    ):
        """
        Generate constraint matrices from the rows of the nullspace basis matrix.
        """
        from popr.base_lifters import StateLifter

        assert isinstance(lifter, StateLifter)

        try:
            n_basis = len(basis)
        except Exception:
            n_basis = basis.shape[0]

        if isinstance(var_dict, list):
            var_dict = lifter.get_var_dict(var_dict)

        A_list = []
        basis_reduced = []
        for i in range(n_basis):
            ai = lifter.get_reduced_a(bi=basis[i], var_subset=var_dict, sparse=True)
            basis_reduced.append(ai)
        basis_reduced = sp.vstack(basis_reduced)

        if AutoTight.REDUCE_DEPENDENT:
            bad_idx = find_dependent_columns(basis_reduced.T, tolerance=1e-6)
        else:
            bad_idx = []

        for i in range(basis_reduced.shape[0]):  # type: ignore
            if i in bad_idx:
                continue
            ai = basis_reduced[[i], :].toarray().flatten()  # type: ignore
            Ai = lifter.get_mat(ai, sparse=sparse, correct=True, var_dict=None)
            # Normalize the matrix
            if normalize and not sparse:
                # Ai /= np.max(np.abs(Ai))
                Ai /= np.linalg.norm(Ai, p=2)  # type: ignore
            elif normalize and sparse:
                Ai /= splinalg.norm(Ai, ord="fro")
            # Sparsify and truncate
            if sparse:
                Ai.eliminate_zeros()  # type: ignore
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0  # type: ignore
            # add to list
            A_list.append(Ai)
        return A_list

    @staticmethod
    def get_duality_gap(cost_local, cost_sdp):
        return (cost_local - cost_sdp) / abs(cost_sdp)
