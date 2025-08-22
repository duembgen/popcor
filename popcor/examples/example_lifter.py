from collections.abc import Iterable

import numpy as np

from popcor.base_lifters import StateLifter


class ExampleLifter(StateLifter):
    """Example Lifter class.

    This class implements the bare minimum to use AutoTight.

    To create a new Lifter for your problem formulation, create
    a copy of this file and fill in the missing parts.

    You can take a look at the :ref:`Examples` for inspiration.
    """

    # choose your homogenization variable here
    HOM = "h"

    # chose the "lifting" levels when going up in the sparse Lasserre's hierarchy.
    LEVELS = ["no"]

    def __init__(self, param_level="no"):
        # you can choose if you want to use parameters. Otherwise remove param_level or set to "no"
        super().__init__(param_level=param_level)

    @property
    def var_dict(self):
        var_dict = {self.HOM: 1}
        return var_dict

    @property
    def param_dict(self):
        param_dict = {self.HOM: 1}
        return param_dict

    def get_x(
        self,
        theta: np.ndarray | None = None,
        var_subset: Iterable | None = None,
        **kwargs
    ) -> np.ndarray:
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            # fill in the rest of x according to var_subset.
            # elif "x" in key:
            # elif "z" in key:
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def sample_parameters(self, theta: np.ndarray | None = None) -> dict:
        return {}

    def sample_theta(self) -> np.ndarray:
        return np.array([])
