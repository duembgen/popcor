"""Poly6Lifter class for sixth-degree polynomial examples."""

import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from popcor.base_lifters import PolyLifter


class Poly6Lifter(PolyLifter):
    """Sixth-degree polynomial examples.

    Two types are provided:

    - poly_type="A": one global minimum, two local minima, two local maxima
    - poly_type="B": one global minimum, one local minimum, one local maximum
    """

    @property
    def VARIABLE_LIST(self) -> list[list[str]]:
        return [[self.HOM, "t", "z0", "z1"]]

    def __init__(self, poly_type: str = "A") -> None:
        assert poly_type in ["A", "B"]
        self.poly_type: str = poly_type
        super().__init__(degree=6)

    def get_Q(
        self, output_poly: bool = False, noise: float | None = None
    ) -> np.ndarray:
        """Returns the Q matrix for the selected polynomial type."""
        if output_poly:
            raise ValueError("output_poly not implemented for Poly6Lifter.")
        if self.poly_type == "A":
            return 0.1 * np.array(
                [
                    [25, 12, 0, 0],
                    [12, -13, -2.5, 0],
                    [0, -2.5, 6.25, -0.9],
                    [0, 0, -0.9, 1 / 6],
                ]
            )
        elif self.poly_type == "B":
            return np.array(
                [
                    [5.0000, 1.3167, -1.4481, 0],
                    [1.3167, -1.4481, 0, 0.2685],
                    [-1.4481, 0, 0.2685, -0.0667],
                    [0, 0.2685, -0.0667, 0.0389],
                ]
            )
        else:
            raise ValueError(f"Unknown poly_type: {self.poly_type}")

    def get_A_known(
        self,
        output_poly: bool = False,
        add_redundant: bool = True,
        var_dict: dict | None = None,
    ) -> tuple[list, list[float]]:
        """Returns the list of known A matrices and their corresponding values."""
        from poly_matrix import PolyMatrix

        A_list: list = []

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1[self.HOM, "z0"] = -1
        A_1["t", "t"] = 2
        A_list.append(A_1)

        # z_1 = t^3 = t z_0
        A_2 = PolyMatrix(symmetric=True)
        A_2[self.HOM, "z1"] = -1
        A_2["t", "z0"] = 1
        A_list.append(A_2)

        # t^4 = z_1 t = z_0 z_0
        if add_redundant:
            B_0 = PolyMatrix(symmetric=True)
            B_0["z0", "z0"] = 2
            B_0["t", "z1"] = -1
            A_list.append(B_0)

        if output_poly:
            return A_list, [0.0] * len(A_list)
        else:
            return [A_i.get_matrix(self.var_dict) for A_i in A_list], [0.0] * len(
                A_list
            )

    def get_D(self, that: float) -> np.ndarray:
        """Returns the D matrix for the given value."""
        D = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [that, 1.0, 0.0, 0.0],
                [that**2, 2 * that, 1.0, 0.0],
                [that**3, 3 * that**2, 3 * that, 1.0],
            ]
        )
        return D

    def generate_random_setup(self) -> None:
        """Initializes a random setup for theta_."""
        self.theta_ = np.array([-1])


if __name__ == "__main__":
    import os

    import matplotlib.pylab as plt

    # Get the directory two levels up from this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    thetas = np.linspace(-1.5, 4.5, 100)
    poly_lifter = Poly6Lifter(poly_type="A")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(os.path.join(base_dir, "docs", "figures", "poly6_lifter_A.png"))

    thetas = np.linspace(-3, 3, 100)
    poly_lifter = Poly6Lifter(poly_type="B")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(os.path.join(base_dir, "docs", "figures", "poly6_lifter_B.png"))

    plt.show(block=False)
    print("done")
