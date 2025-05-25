"""
The BaseClass contains all the functionalities that are required by StateLifter but that
are tedious and uninteresting, such as converting between different formats.

Ideally, the user never has to look at this code.
"""

import itertools
from abc import abstractmethod
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix, unroll

from popr.utils.common import create_symmetric, get_labels, get_vec
from popr.utils.constraint import Constraint, remove_dependent_constraints


class BaseClass(object):
    # Homogenization variable name
    HOM = "h"

    # set elements below this threshold to zero.
    EPS_SPARSE = 1e-9

    # properties of template scaling
    ALL_PAIRS = True
    # Below only have effect if ALL_PAIRS is False.
    # Then, they determine the clique size hierarchy.
    CLIQUE_SIZE = 5
    STEP_SIZE = 1

    # tolerance for feasibility error of learned constraints
    EPS_ERROR = 1e-8

    PARAM_LEVELS = ["no", "p", "ppT"]

    @property
    @abstractmethod
    def var_dict(self) -> dict:
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        pass

    @abstractmethod
    def sample_theta(self) -> np.ndarray:
        pass

    @abstractmethod
    def sample_parameters(self) -> np.ndarray:
        pass

    @staticmethod
    def get_variable_indices(var_subset, variable="z"):
        return np.unique(
            [int(v.split("_")[-1]) for v in var_subset if v.startswith(f"{variable}_")]
        )

    def __init__(self, d, param_level, n_parameters):
        assert param_level in self.PARAM_LEVELS
        self.param_level = param_level
        self.d = d

        if param_level == "no":
            self.n_parameters = 1
        elif param_level in ["p", "ppT"]:
            self.n_parameters = n_parameters
        else:
            raise ValueError(f"Unknown param_level: {param_level}")
        # n_parameters * (self.d + (self.d * (self.d + 1)) // 2)
        self.generate_random_setup()

    ### Functionalities related to var_dict
    def get_var_dict(self, var_subset=None, unroll_keys=False):
        if var_subset is not None:
            var_dict = {k: v for k, v in self.var_dict.items() if k in var_subset}
            if unroll_keys:
                return unroll(var_dict)
            else:
                return var_dict
        if unroll_keys:
            return unroll(self.var_dict)
        return self.var_dict

    def get_param_dict(self, param_subset=None):
        if param_subset is not None:
            return {k: v for k, v in self.param_dict.items() if k in param_subset}
        return self.param_dict

    ### Functionalities related to random setups
    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    @theta.setter
    def theta(self, t):
        assert (
            self.theta_ is None
        ), "The property self.theta is only meant to be set once!"
        self.theta_ = t

    @property
    def parameters(self) -> dict:
        if self.parameters_ is None:
            self.parameters_ = self.sample_parameters()
            assert isinstance(self.parameters_, dict)
        return self.parameters_  # type: ignore

    @parameters.setter
    def parameters(self, p):
        assert (
            self.parameters_ is None
        ), "The property self.parameters is only meant to be set once!"
        assert isinstance(p, dict)
        self.parameters_ = p

    def extract_A_known(self, A_known, var_subset, output_type="csc"):
        """
        Extract from the list of constraint matrices only the ones that
        touch only a subset of var_subset.
        """
        if output_type == "dense":
            sub_A_known = np.empty((0, self.get_dim_Y(var_subset)))
        else:
            sub_A_known = []
        for A in A_known:
            A_poly, var_dict = PolyMatrix.init_from_sparse(A, self.var_dict)

            assert len(A_poly.get_variables()) > 0

            # if all of the non-zero elements of A_poly are in var_subset,
            # we can use this matrix.
            if np.all([v in var_subset for v in A_poly.get_variables()]):
                Ai = A_poly.get_matrix(
                    self.get_var_dict(var_subset), output_type=output_type
                )
                if output_type == "dense":
                    ai = self.augment_using_zero_padding(
                        get_vec(Ai, correct=True), var_subset
                    )
                    sub_A_known = np.r_[sub_A_known, ai[None, :]]
                else:
                    assert isinstance(sub_A_known, list)
                    sub_A_known.append(Ai)
        return sub_A_known

    def get_param_idx_dict(self, var_subset=None):
        """
        Give the current subset of variables, extract the parameter dictionary to use.
        Example:
            var_subset = ['z_0', 'x_1']
            -> Parameters to include:  self.HOM (always), p_0, p_1
        - if param_level == 'no': {'l': 0}
        - if param_level == 'p': {'l': 0, 'p_0:0': 1, ..., 'p_0:d-1': d}
        - if param_level == 'ppT': {'l': 0, 'p_0:0.p_0:0': 1, ..., 'p_0:d-1:.p_0:d-1': 1}
        """
        # TODO(FD) change to use  param_subset instead.
        param_subset = [self.HOM] + [
            f"p_{i}" for i in self.get_variable_indices(var_subset)
        ]
        return unroll(self.get_param_dict(param_subset))

    def get_mat(self, vec, sparse=False, var_dict=None, correct=True):
        """Convert (N+1)N/2 vectorized matrix to NxN Symmetric matrix in a way that preserves inner products.

        In particular, this means that we divide the off-diagonal elements by sqrt(2).

        :param vec (ndarray): vector of upper-diagonal elements
        :return: symmetric matrix filled with vec.
        """
        # len(vec) = k = n(n+1)/2 -> dim_x = n =
        if var_dict is None:
            pass

        elif not isinstance(var_dict, dict):
            var_dict = {k: v for k, v in self.var_dict.items() if k in var_dict}

        Ai = create_symmetric(
            vec, correct=correct, sparse=sparse, eps_sparse=self.EPS_SPARSE
        )
        if var_dict is None:
            return Ai

        # if var_dict is not None, then Ai corresponds to the subblock
        # defined by var_dict, of the full constraint matrix.
        Ai_poly, __ = PolyMatrix.init_from_sparse(Ai, var_dict, unfold=True)
        from poly_matrix.poly_matrix import augment

        augment_var_dict = augment(self.var_dict)
        all_var_dict = {key[2]: 1 for key in augment_var_dict.values()}
        return Ai_poly.get_matrix(all_var_dict)

    def var_list_row(
        self, var_subset=None, param_subset=None, force_parameters_off=False
    ):
        if var_subset is None:
            var_subset = list(self.var_dict.keys())
        elif isinstance(var_subset, dict):
            var_subset = list(var_subset.keys())

        if param_subset is None:
            param_subset = self.param_dict

        label_list = []
        if force_parameters_off:
            param_dict = {self.HOM: 0}
        else:
            param_dict = unroll(
                self.get_param_dict(param_subset)
            )  # self.get_param_idx_dict(var_subset)

        for idx, key in enumerate(param_dict.keys()):
            for i in range(len(var_subset)):
                zi = var_subset[i]
                sizei = self.var_dict[zi]
                for di in range(sizei):
                    keyi = f"{zi}:{di}" if sizei > 1 else f"{zi}"
                    for j in range(i, len(var_subset)):
                        zj = var_subset[j]
                        sizej = self.var_dict[zj]
                        if zi == zj:
                            djs = range(di, sizej)
                        else:
                            djs = range(sizej)

                        for dj in djs:
                            keyj = f"{zj}:{dj}" if sizej > 1 else f"{zj}"
                            label_list.append(f"{key}-{keyi}.{keyj}")
            # for zi, zj in vectorized_var_list:
            # label_list += self.get_labels(key, zi, zj)
            assert len(label_list) == (idx + 1) * self.get_dim_X(var_subset)
        return label_list

    def var_dict_row(self, var_subset=None, force_parameters_off=False):
        return {
            label: 1
            for label in self.var_list_row(
                var_subset, force_parameters_off=force_parameters_off
            )
        }

    def get_basis_from_poly_rows(self, basis_poly_list, var_subset=None):
        var_dict = self.get_var_dict(var_subset=var_subset)

        all_dict = {label: 1 for label in self.var_list_row(var_subset)}
        basis_reduced = np.empty((0, len(all_dict)))
        for i, bi_poly in enumerate(basis_poly_list):
            # test that this constraint holds

            bi = bi_poly.get_matrix(([self.HOM], all_dict))

            if bi.shape[1] == self.get_dim_X(var_subset) * self.get_dim_P():
                ai = self.get_reduced_a(bi, var_subset=var_subset)
                Ai = self.get_mat(ai, var_dict=var_dict)
            elif bi.shape[1] == self.get_dim_X():
                Ai = self.get_mat(bi, var_dict=var_subset)

            err, idx = self.test_constraints([Ai], errors="print")
            if len(idx):
                print(f"found b{i} has error: {err}")
                continue

            # test that this constraint is lin. independent of previous ones.
            basis_reduced_test = np.vstack([basis_reduced, bi.toarray()])
            rank = np.linalg.matrix_rank(basis_reduced_test)
            if rank == basis_reduced_test.shape[0]:
                basis_reduced = basis_reduced_test
            else:
                print(f"b{i} is linearly dependant after factoring out parameters.")
        print(f"left with {basis_reduced.shape} total constraints")
        return basis_reduced

    def convert_polyrow_to_Apoly(self, poly_row, correct=True):
        """Convert poly-row to reduced a.

        poly-row has elements with keys "pk:l.xi:m.xj:n",
        meaning this element corresponds to the l-th element of parameter i,
        and the m-n-th element of xj times xk.
        """
        parameters = self.get_p()
        param_dict = dict(zip(unroll(self.param_dict), parameters))

        poly_mat = PolyMatrix(symmetric=True)
        for key in poly_row.variable_dict_j:
            param, var_keys = key.split("-")
            param_val = param_dict[param]

            keyi_m, keyj_n = var_keys.split(".")
            m = keyi_m.split(":")[-1]
            n = keyj_n.split(":")[-1]

            # divide off-diagonal elements by sqrt(2)
            newval = poly_row[self.HOM, key] * param_val
            if correct and not ((keyi_m == keyj_n) and (m == n)):
                newval /= np.sqrt(2)

            poly_mat[keyi_m, keyj_n] += newval
        return poly_mat

    def convert_polyrow_to_Asparse(self, poly_row, var_subset=None):
        poly_mat = self.convert_polyrow_to_Apoly(poly_row, correct=False)

        var_dict = self.get_var_dict(var_subset)
        mat_var_list = []
        for var, size in var_dict.items():
            if size == 1:
                mat_var_list.append(var)
            else:
                mat_var_list += [f"{var}:{i}" for i in range(size)]
        mat_sparse = poly_mat.get_matrix({m: 1 for m in mat_var_list})
        return mat_sparse

    def convert_polyrow_to_a(self, poly_row, var_subset=None, sparse=False):
        """Convert poly-row to reduced a.

        poly-row has elements with keys "pk:l.xi:m.xj:n",
        meaning this element corresponds to the l-th element of parameter i,
        and the m-n-th element of xj times xk.
        """
        mat_sparse = self.convert_polyrow_to_Asparse(poly_row, var_subset)
        return get_vec(mat_sparse, correct=False, sparse=sparse)

    # TODO(FD) consider removing this cause only used in tests.
    def convert_a_to_polyrow(
        self,
        a,
        var_subset=None,
    ) -> PolyMatrix:
        """Convert a array to poly-row."""
        if var_subset is None:
            var_subset = self.var_dict
        var_dict = self.get_var_dict(var_subset)
        dim_X = self.get_dim_X(var_subset)

        try:
            dim_a = len(a)
        except Exception:
            dim_a = a.shape[1]
        assert dim_a == dim_X

        mat = create_symmetric(a, sparse=True, eps_sparse=self.EPS_SPARSE)
        poly_mat, __ = PolyMatrix.init_from_sparse(mat, var_dict)
        poly_row = PolyMatrix(symmetric=False)
        for keyi, keyj in itertools.combinations_with_replacement(var_dict, 2):
            if keyi in poly_mat.matrix and keyj in poly_mat.matrix[keyi]:
                val = poly_mat.matrix[keyi][keyj]
                labels = get_labels(self.HOM, keyi, keyj, self.var_dict)
                if keyi != keyj:
                    vals = val.flatten()
                else:
                    # TODO: use get_vec instead?
                    vals = val[np.triu_indices(val.shape[0])]
                assert len(labels) == len(vals)
                for label, v in zip(labels, vals):
                    if np.any(np.abs(v) > self.EPS_SPARSE):
                        poly_row[self.HOM, label] = v
        return poly_row

    def convert_b_to_polyrow(
        self, b, var_subset, param_subset=None, tol=1e-10
    ) -> PolyMatrix:
        """Convert (augmented) b array to poly-row."""
        if isinstance(b, PolyMatrix):
            raise NotImplementedError(
                "can't call convert_b_to_polyrow with PolyMatrix yet."
            )

        assert len(b) == self.get_dim_Y(var_subset, param_subset)
        poly_row = PolyMatrix(symmetric=False)
        mask = np.abs(b) > tol

        # get the variable names such as p_0:0-x:0.x:4 whch corresponds to p_0[0]*x[0]*x[4]
        var_list_row = self.var_list_row(var_subset, param_subset)
        assert len(b) == len(var_list_row)

        for idx in np.where(mask == True)[0]:
            poly_row[self.HOM, var_list_row[idx]] = b[idx]
        return poly_row

    def get_dim_x(self, var_subset=None):
        var_dict = self.get_var_dict(var_subset)
        return sum([val for val in var_dict.values()])

    def get_dim_p(self, param_subset=None):
        param_dict = self.get_param_dict(param_subset)
        return sum([val for val in param_dict.values()])

    def get_dim_Y(self, var_subset=None, param_subset=None):
        dim_X = self.get_dim_X(var_subset=var_subset)
        dim_P = self.get_dim_P(param_subset=param_subset)
        return int(dim_X * dim_P)

    def get_dim_X(self, var_subset=None):
        dim_x = self.get_dim_x(var_subset)
        return int(dim_x * (dim_x + 1) / 2)

    def get_dim_P(self, param_subset=None):
        return len(self.get_p(param_subset=param_subset))

    def get_reduced_a(
        self, bi, param_here=None, var_subset=None, param_subset=None, sparse=False
    ):
        """
        Extract first block of bi by summing over other blocks times the parameters.
        """
        if param_here is None:
            param_here = self.get_p(param_subset=param_subset)

        if isinstance(bi, np.ndarray):
            len_b = len(bi)
        elif isinstance(bi, PolyMatrix):
            bi = bi.get_matrix(([self.HOM], self.var_dict_row(var_subset)))
            len_b = bi.shape[1]  # type: ignore
        else:
            # bi can be a scipy sparse matrix,
            len_b = bi.shape[1]

        n_params = self.get_dim_P(param_subset=param_subset)
        dim_X = self.get_dim_X(var_subset)
        n_parts = len_b / dim_X
        assert (
            n_parts == n_params
        ), f"{len_b} does not not split in dim_P={n_params} parts of size dim_X={dim_X}"

        ai = np.zeros(dim_X)
        for i, p in enumerate(param_here):
            if isinstance(bi, np.ndarray):
                ai += p * bi[i * dim_X : (i + 1) * dim_X]
            elif isinstance(bi, sp.csc_array):
                ai += p * bi[0, i * dim_X : (i + 1) * dim_X].toarray().flatten()
            else:
                raise ValueError("untreated case:", type(bi))
        if sparse:
            ai_sparse = sp.csr_array(ai[None, :])
            ai_sparse.eliminate_zeros()
            return ai_sparse
        else:
            return ai

    def augment_using_zero_padding(self, ai, param_subset=None):
        n_parameters = self.get_dim_P(param_subset)
        return np.hstack([ai, np.zeros((n_parameters - 1) * len(ai))])

    def augment_using_parameters(self, x, param_subset=None):
        p = self.get_p(param_subset)
        return np.kron(p, x)

    def compute_Ai(self, templates, var_dict, param_dict):
        """
        Take all elements from the list of templates and apply them
        to the given pair of var_list and param_list.
        """

        A_list = []
        for k, template in enumerate(templates):
            assert template.b_ is not None
            assert isinstance(template, Constraint)
            # First, we find the current parameters, so that we can factor
            # them into b and compute a from it.
            p_here = self.get_p(param_subset=param_dict)

            # We need to partition the vector b into its subblocks
            # so that we can compute a from it.
            X_dim = self.get_dim_X(template.mat_var_dict)
            assert self.get_dim_X(var_dict) == X_dim
            p_dim = self.get_dim_p(template.mat_param_dict)
            assert self.get_dim_p(param_dict) == p_dim
            n_blocks = int(len(template.b_) / X_dim)
            assert n_blocks == p_dim

            a = p_here[0] * template.b_[:X_dim]
            for i in range(n_blocks - 1):
                a += p_here[i + 1] * template.b_[(i + 1) * X_dim : (i + 2) * X_dim]

            if not np.any(np.abs(a) > self.EPS_SPARSE):
                print(f"matrix {k} is zero")
                continue

            a_test = self.get_reduced_a(
                template.b_,
                var_subset=template.mat_var_dict,
                param_subset=template.mat_param_dict,
                param_here=p_here,
                sparse=False,
            )
            assert isinstance(a_test, np.ndarray)
            np.testing.assert_allclose(a, a_test)

            # Get a symmetric matrix where the upper and lower parts have been filled with a,
            # and applying the correction to the diagonal.
            # Note that we do not set var_dict because otherwise A would already
            # be the zero-padded large matrix.
            A = self.get_mat(a, sparse=True, correct=True)

            # Get the corresponding PolyMatrix.
            A_poly, __ = PolyMatrix.init_from_sparse(
                A,
                var_dict=self.get_var_dict(var_dict),
                symmetric=True,
                unfold=False,
            )
            A_list.append(A_poly)
        return A_list

    def get_constraint_rank(self, A_list_poly, output_sorted=False):
        """Find the number of independent constraints when they are of the form A_i @ x = 0.

        :param A_list_poly: list of constraints matrices

        :return: rank (int) and sorted matrices (list, if output_sorted=True) where the first rank matrices correspond to
                 linearly independent matrices.
        """
        x = self.get_x()

        current_rank = 0
        independent_indices = []
        dependent_indices = []
        basis_incremental = np.zeros((len(x), 0))
        for i, Ai in enumerate(A_list_poly):
            if isinstance(Ai, PolyMatrix):
                new_candidate = (Ai.get_matrix(self.var_dict) @ x).reshape((-1, 1))
            else:
                new_candidate = (Ai @ x).reshape((-1, 1))
            basis_candidate = np.hstack([basis_incremental, new_candidate])
            new_rank = np.linalg.matrix_rank(basis_candidate)
            if new_rank == current_rank + 1:
                independent_indices.append(i)
                basis_incremental = basis_candidate
                current_rank = new_rank
            else:
                dependent_indices.append(i)
        if not output_sorted:
            return current_rank
        A_list_sorted = [A_list_poly[i] for i in independent_indices] + [
            A_list_poly[i] for i in dependent_indices
        ]
        return current_rank, A_list_sorted

    def apply_template(self, bi_poly, n_parameters=None, verbose=False):
        if n_parameters is None:
            n_parameters = len(self.parameters)

        new_poly_rows = []
        # find the number of variables that this constraint touches.
        unique_idx = set()
        for key in bi_poly.variable_dict_j:
            param, var_keys = key.split("-")
            vars = var_keys.split(".")
            vars += param.split(".")
            for var in vars:
                var_base = var.split(":")[0]
                if "_" in var_base:
                    i = int(var_base.split("_")[-1])
                    unique_idx.add(i)

        if len(unique_idx) == 0:
            return [bi_poly]
        elif len(unique_idx) > 3:
            raise ValueError("unexpected triple dependencies!")

        variable_indices = self.get_variable_indices(self.var_dict)
        # if z_0 is in this constraint, repeat the constraint for each landmark.
        for idx in itertools.combinations(variable_indices, len(unique_idx)):
            new_poly_row = PolyMatrix(symmetric=False)
            for key in bi_poly.variable_dict_j:
                # need intermediate variables cause otherwise z_0 -> z_1 -> z_2 etc. can happen.
                key_ij = key
                for from_, to_ in zip(unique_idx, idx):
                    key_ij = key_ij.replace(f"x_{from_}", f"xi_{to_}")
                    key_ij = key_ij.replace(f"w_{from_}", f"wi_{to_}")
                    key_ij = key_ij.replace(f"z_{from_}", f"zi_{to_}")
                    key_ij = key_ij.replace(f"p_{from_}", f"pi_{to_}")
                key_ij = (
                    key_ij.replace("zi", "z")
                    .replace("pi", "p")
                    .replace("xi", "x")
                    .replace("wi", "w")
                )
                if verbose and (key != key_ij):
                    print("changed", key, "to", key_ij)

                try:
                    params = key_ij.split("-")[0]
                    pi, pj = params.split(".")
                    pi, di = pi.split(":")
                    pj, dj = pj.split(":")
                    if pi == pj:
                        if not (int(dj) >= int(di)):
                            raise IndexError(
                                "something went wrong in augment_basis_list"
                            )
                except ValueError as e:
                    pass
                new_poly_row[self.HOM, key_ij] = bi_poly[self.HOM, key]
            new_poly_rows.append(new_poly_row)
        return new_poly_rows

    def apply_templates(
        self, templates, starting_index=0, var_dict=None, all_pairs=None
    ):

        if all_pairs is None:
            all_pairs = self.ALL_PAIRS
        if var_dict is None:
            var_dict = self.var_dict

        new_constraints = []
        index = starting_index
        for template in templates:
            constraints = self.apply_template(template.polyrow_b_)
            template.applied_list = []
            for new_constraint in constraints:
                template.applied_list.append(
                    Constraint.init_from_polyrow_b(
                        index=index,
                        polyrow_b=new_constraint,
                        lifter=self,
                        template_idx=template.index,
                        known=template.known,
                        mat_var_dict=var_dict,
                    )
                )
                new_constraints += template.applied_list
                index += 1

        if len(new_constraints):
            remove_dependent_constraints(new_constraints)
        return new_constraints

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def test_constraints(self, A_list, errors: str = "raise", n_seeds: int = 3):
        """
        :param A_list: can be either list of sparse matrices, or poly matrices
        :param errors: "raise" or "print" detected violations.
        """
        max_violation = -np.inf
        j_bad = set()

        for j, A in enumerate(A_list):
            if isinstance(A, PolyMatrix):
                A = A.get_matrix(self.get_var_dict(unroll_keys=True))

            for i in range(n_seeds):
                if i == 0:
                    x = self.get_x().flatten()
                else:
                    np.random.seed(i)
                    t = self.sample_theta()
                    p = self.sample_parameters()
                    x = self.get_x(theta=t, parameters=p).flatten()

                constraint_violation = abs(float(x.T @ A @ x))
                max_violation = max(max_violation, constraint_violation)
                if constraint_violation > self.EPS_ERROR:
                    msg = f"big violation at {j}: {constraint_violation:.1e}"
                    j_bad.add(j)
                    if errors == "raise":
                        raise ValueError(msg)
                    elif errors == "print":
                        print(msg)
                    elif errors == "ignore":
                        pass
                    else:
                        raise ValueError(errors)
        return max_violation, j_bad

    def get_A0(self, var_subset=None):
        if var_subset is not None:
            var_dict = {k: self.var_dict[k] for k in var_subset}
        else:
            var_dict = self.var_dict
        A0 = PolyMatrix()
        A0[self.HOM, self.HOM] = 1.0
        return A0.get_matrix(var_dict)

    def get_A_b_list(self, A_list, var_subset=None):
        return [(self.get_A0(var_subset), 1.0)] + [(A, 0.0) for A in A_list]

    def sample_parameters_landmarks(self, landmarks: np.ndarray):
        """Used by RobustPoseLifter, RangeOnlyLocLifter: the default way of adding landmarks to parameters."""
        parameters = {self.HOM: 1.0}
        n_landmarks = landmarks.shape[0]

        if self.param_level == "no":
            return parameters

        for i in range(n_landmarks):
            if self.param_level == "p":
                parameters[f"p_{i}"] = landmarks[i]
            elif self.param_level == "ppT":
                parameters[f"p_{i}"] = np.hstack(  # type: ignore
                    [
                        landmarks[i],
                        get_vec(  # type: ignore
                            np.outer(landmarks[i], landmarks[i]),
                            correct=False,
                            sparse=False,
                        ),
                    ]
                )
        return parameters

    def get_error(self, t) -> dict:
        err = np.linalg.norm(t - self.theta) ** 2 / self.theta.size
        return {"MSE": err, "error": err}

    def generate_random_setup(self):
        if self.parameters is None:
            self.parameters = self.sample_parameters()
        if self.theta is None:
            self.theta = self.sample_theta()

    def get_x(
        self,
        theta: np.ndarray | None = None,
        var_subset: Iterable | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Get the lifted state vector x.

        :param theta: if given, use this theta instead of the ground truth one.
        :param var_subset: list of parameter keys to use. If None, use all.

        :returns: lifted vector x
        """
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict

        assert theta is not None

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            else:
                print(
                    "Warning: just using theta in x because there is no specific implementation."
                )
                x_data += list(theta)
        return np.hstack(x_data)

    def get_p(
        self, parameters: dict | None = None, param_subset: dict | list | None = None
    ):
        if parameters is None:
            parameters = self.parameters
        if param_subset is None:
            param_subset = self.param_dict

        assert isinstance(parameters, dict)

        p_data = []
        for key in param_subset:
            if key == self.HOM:
                p_data.append(1.0)
            else:
                param = parameters[key]
                if np.ndim(param) == 0:
                    p_data.append(param)
                else:
                    p_data += list(param)
        return np.array(p_data)
