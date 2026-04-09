"""Poly6Lifter class for sixth-degree polynomial examples."""

import warnings

import numpy as np
from poly_matrix import PolyMatrix

from popcor.base_lifters import PolyLifter


class Poly6Lifter(PolyLifter):
    """Sixth-degree polynomial examples.

    Two example types are provided and selected with create_example(example_type=...).

    - example_type="A": one global minimum, two local minima, two local maxima
    - example_type="B": one global minimum, one local minimum, one local maximum
    """

    EXAMPLE_TYPES: tuple[str, str] = ("A", "B")
    example_type: str = "A"

    @property
    def VARIABLE_LIST(self) -> list[list[str]]:
        return [[self.HOM, "t", "z0", "z1"]]

    def __init__(self) -> None:
        self.example_type = "A"
        super().__init__(degree=6)

    @staticmethod
    def create_example(example_type: str = "A") -> "Poly6Lifter":
        """Create a sixth-degree polynomial example of the requested type."""
        if example_type not in Poly6Lifter.EXAMPLE_TYPES:
            raise ValueError(
                f"Unknown example_type: {example_type}. Expected one of {Poly6Lifter.EXAMPLE_TYPES}."
            )

        lifter = Poly6Lifter()
        lifter.example_type = example_type
        lifter.generate_random_setup()
        return lifter

    def get_Q(
        self, output_poly: bool = False, noise: float | None = None
    ) -> np.ndarray:
        """Return the Q matrix for the selected example type."""
        if output_poly:
            raise ValueError("output_poly not implemented for Poly6Lifter.")
        if self.example_type == "A":
            return 0.1 * np.array(
                [
                    [25, 12, 0, 0],
                    [12, -13, -2.5, 0],
                    [0, -2.5, 6.25, -0.9],
                    [0, 0, -0.9, 1 / 6],
                ]
            )
        elif self.example_type == "B":
            return np.array(
                [
                    [5.0000, 1.3167, -1.4481, 0],
                    [1.3167, -1.4481, 0, 0.2685],
                    [-1.4481, 0, 0.2685, -0.0667],
                    [0, 0.2685, -0.0667, 0.0389],
                ]
            )
        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def get_A_known(
        self,
        output_poly: bool = False,
        add_redundant: bool = True,
        var_dict: dict | None = None,
    ) -> tuple[list, list[float]]:
        """Returns the list of known A matrices and their corresponding values."""
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

    def plot_cost(
        self,
        y: np.ndarray | None = None,
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
        grid_size: int = 120,
        thetas: np.ndarray | None = None,
        label: str | None = None,
    ) -> tuple[object, object, object]:
        """Plot cost using the same signature as range-only lifters."""
        if y is not None:
            warnings.warn(
                "y is ignored in plot_cost for Poly6Lifter.",
                RuntimeWarning,
                stacklevel=2,
            )
        if thetas is not None:
            warnings.warn(
                "thetas is ignored in plot_cost for Poly6Lifter.",
                RuntimeWarning,
                stacklevel=2,
            )
        if label is not None:
            warnings.warn(
                "label is ignored in plot_cost for Poly6Lifter.",
                RuntimeWarning,
                stacklevel=2,
            )

        theta_ref = float(np.asarray(self.theta).reshape(-1)[0])
        if xlims is None:
            xlims = (theta_ref - 3.0, theta_ref + 4.0)

        thetas = np.linspace(xlims[0], xlims[1], grid_size)
        fig, ax = self.plot(thetas, label="cost", estimates={"theta_ref": theta_ref})
        line = ax.lines[-1] if len(ax.lines) else None
        if ylims is not None:
            ax.set_ylim(*ylims)
        ax.legend()
        return fig, ax, line

    def generate_random_setup(self) -> None:
        """Initializes a random setup for theta_."""
        self.theta_ = np.array([-1])


if __name__ == "__main__":
    import os

    import matplotlib.pylab as plt

    # Get the directory two levels up from this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    thetas = np.linspace(-1.5, 4.5, 100)
    poly_lifter = Poly6Lifter.create_example(example_type="A")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(
        os.path.join(base_dir, "docs", "source", "_static", "poly6_lifter_A.png")
    )

    thetas = np.linspace(-3, 3, 100)
    poly_lifter = Poly6Lifter.create_example(example_type="B")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(
        os.path.join(base_dir, "docs", "source", "_static", "poly6_lifter_B.png")
    )

    plt.show(block=False)
    print("done")
