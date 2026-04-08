"""Poly4Lifter class for fourth-degree polynomial lifter examples."""

import numpy as np
from poly_matrix import PolyMatrix

from popcor.base_lifters import PolyLifter


class Poly4Lifter(PolyLifter):
    """Fourth-degree polynomial lifter.

    Two example types are provided and selected with create_example(example_type=...).

    - example_type="A": one global minimum at -1, one local minimum at 2.
    - example_type="B": two global minima at 1 and 3.
    """

    EXAMPLE_TYPES: tuple[str, str] = ("A", "B")
    example_type: str = "A"

    @property
    def VARIABLE_LIST(self) -> list[list[str]]:
        return [[self.HOM, "t", "z0"]]

    def __init__(self) -> None:
        self.example_type = "A"
        super().__init__(degree=4)

    @staticmethod
    def create_example(example_type: str = "A") -> "Poly4Lifter":
        """Create a fourth-degree polynomial example of the requested type."""
        if example_type not in Poly4Lifter.EXAMPLE_TYPES:
            raise ValueError(
                f"Unknown example_type: {example_type}. Expected one of {Poly4Lifter.EXAMPLE_TYPES}."
            )

        lifter = Poly4Lifter()
        lifter.example_type = example_type
        lifter.generate_random_setup()
        return lifter

    def get_Q(
        self, output_poly: bool = False, noise: float | None = None
    ) -> np.ndarray:
        """Return the Q matrix for the selected example type."""
        if output_poly:
            raise ValueError("output_poly not implemented for Poly4Lifter.")
        if self.example_type == "A":
            # Q matrix for type A
            return np.r_[
                np.c_[2, 1, 0], np.c_[1, -1 / 2, -1 / 3], np.c_[0, -1 / 3, 1 / 4]
            ]
        elif self.example_type == "B":
            # Q matrix for type B, constructed such that f'(t) = (t-1)*(t-2)*(t-3)
            return np.r_[np.c_[3, -3, 0], np.c_[-3, 11 / 2, -1], np.c_[0, -1, 1 / 4]]
        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def get_A_known(
        self,
        output_poly: bool = False,
        add_redundant: bool = False,
        var_dict: dict | None = None,
    ) -> tuple[list[np.ndarray] | list, list[int]]:
        if add_redundant:
            print("No redundant constraints for 4-degree polynomial.")

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1[self.HOM, "z0"] = -1
        A_1["t", "t"] = 2
        if output_poly:
            return [A_1], [0]
        else:
            return [A_1.get_matrix(self.var_dict)], [0]

    def generate_random_setup(self) -> None:
        if self.example_type == "A":
            self.theta_: np.ndarray = np.array([-1])
        elif self.example_type == "B":
            self.theta_: np.ndarray = np.array([1])
        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def get_D(self, that: float) -> np.ndarray:
        D = np.array(
            [
                [1.0, 0.0, 0.0],
                [that, 1.0, 0.0],
                [that**2, 2 * that, 1.0],
            ]
        )
        return D


if __name__ == "__main__":
    import os

    import matplotlib.pylab as plt

    # Get the directory two levels up from this file
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    thetas: np.ndarray = np.linspace(-2, 3, 100)
    poly_lifter: Poly4Lifter = Poly4Lifter.create_example(example_type="A")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(
        os.path.join(base_dir, "docs", "source", "_static", "poly4_lifter_A.png")
    )

    thetas = np.linspace(0, 4, 100)
    poly_lifter = Poly4Lifter.create_example(example_type="B")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(
        os.path.join(base_dir, "docs", "source", "_static", "poly4_lifter_B.png")
    )

    plt.show(block=False)
    print("done")
