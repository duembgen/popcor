"""Example module for implementing a custom StateLifter.

This module provides an ExampleLifter class that demonstrates the minimal implementation required to use AutoTight.
"""

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

    HOM: str = "h"
    LEVELS: list[str] = ["no"]

    def __init__(self, param_level: str = "no") -> None:
        super().__init__(param_level=param_level)

    @property
    def var_dict(self) -> dict[str, int]:
        var_dict = {self.HOM: 1}
        return var_dict

    @property
    def param_dict(self) -> dict[str, int]:
        param_dict = {self.HOM: 1}
        return param_dict

    def get_x(
        self,
        theta: np.ndarray | None = None,
        var_subset: dict | list | None = None,
        parameters: np.ndarray | dict | None = None,
        **kwargs
    ) -> np.ndarray:
        """Returns the lifted variable vector x for the given subset."""
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict

        x_data: list[float] = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)

            # possibly use the parameter, if x depends on it.
            if isinstance(parameters, dict):
                pi = parameters[key]
            else:
                pi = parameters

            # fill in the rest of x according to var_subset.
            # elif "x" in key:
            # elif "z" in key:

        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def sample_parameters(self, theta: np.ndarray | None = None) -> dict:
        """Samples parameters for the lifter."""
        return {}

    def sample_theta(self) -> np.ndarray:
        """Samples theta for the lifter."""
        return np.array([])
