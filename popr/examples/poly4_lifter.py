import numpy as np

from popr.base_lifters import PolyLifter


class Poly4Lifter(PolyLifter):
    """Fourth-degree polynomial examples.

    Two types are provided:

    - poly_type="A": one global minimum, one local minimum
    - poly_type="B": two global minima
    """

    @property
    def VARIABLE_LIST(self):
        return [[self.HOM, "t", "z0"]]

    def __init__(self, poly_type="A"):
        # actual minimum
        assert poly_type in ["A", "B"]
        self.poly_type = poly_type
        super().__init__(degree=4)

    def get_Q(self, *args, **kwargs):
        if self.poly_type == "A":
            # fmt: off
            Q = np.r_[
                np.c_[2, 1, 0], 
                np.c_[1, -1 / 2, -1 / 3], 
                np.c_[0, -1 / 3, 1 / 4]
            ]
            # fmt: on
        elif self.poly_type == "B":
            # below is constructed such that f'(t) = (t-1)*(t-2)*(t-3)
            # fmt: off
            Q = np.r_[
                np.c_[3, -3, 0], 
                np.c_[-3, 11/2, -1], 
                np.c_[0, -1, 1/4]
            ]
            # fmt: on
        return Q

    def get_A_known(self, output_poly=False, add_redundant=False):
        from poly_matrix import PolyMatrix

        if add_redundant:
            print("No redundant constraitns for 4-degree polynomial.")

        # z_0 = t^2
        A_1 = PolyMatrix(symmetric=True)
        A_1[self.HOM, "z0"] = -1
        A_1["t", "t"] = 2
        if output_poly:
            return [A_1]
        else:
            return [A_1.get_matrix(self.var_dict)]

    def generate_random_setup(self):
        self.theta_ = np.array([-1])

    def get_D(self, that):
        """Not currently used."""
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
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    thetas = np.linspace(-2, 3, 100)
    poly_lifter = Poly4Lifter(poly_type="A")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(os.path.join(base_dir, "docs", "figures", "poly4_lifter_A.png"))

    thetas = np.linspace(0, 4, 100)
    poly_lifter = Poly4Lifter(poly_type="B")
    fig, ax = poly_lifter.plot(thetas)
    fig.savefig(os.path.join(base_dir, "docs", "figures", "poly4_lifter_B.png"))

    plt.show(block=False)
    print("done")
