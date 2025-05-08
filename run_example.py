import matplotlib.pylab as plt
import numpy as np
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from scipy.optimize import minimize

from popr.examples import Poly4Lifter, Poly6Lifter
from popr.utils.plotting_tools import savefig

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.size": 12,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

SAVEFIG = False

def coordinate_system():
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    ax.yaxis.set_ticks_position("left")
    # make arrows
    ax.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=5,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=5,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.set_xlabel("$\\theta$", loc="right")
    ax.set_ylabel("$f(\\theta)$", loc="top", rotation=0)
    ax.set_yticks([])
    return fig, ax


def plot_cost(ts, y, label=None):
    fig, ax = coordinate_system()
    ax.plot(ts, y, label=label)
    ymin = min(-max(y) / 3, min(y))
    ax.set_ylim(ymin, max(y))
    return fig, ax


def plot_global(lifter, ax, sol_dict):
    import itertools

    markers = itertools.cycle(["x", "+", "o", "*"])
    for label, t in sol_dict.items():
        ax.scatter(
            [t],
            [lifter.get_cost(t)],
            color="C2",
            marker=next(markers),
            label=label,
        )
    ax.legend(loc="upper center")


if __name__ == "__main__":

    lifter4 = Poly4Lifter()
    lifter4.xlims = [-2, 3.1]
    lifter4.solution_list = [
        dict(x0=-1.9, label="global min", name="global min", color="C2"),
        dict(x0=3.1, label="local min", name="local min", color="C3"),
        dict(x0=0.0, label="local max", name="local max", color="C0"),
    ]

    lifter6 = Poly6Lifter(poly_type="A")
    lifter6.xlims = [-2, 5.1]
    lifter6.solution_list = [
        dict(x0=-2, label="global min", name="global min", color="C2"),
        dict(x0=2.5, label="local min", name="local min", color="C3"),
        dict(x0=5, label=None, name="local min", color="C3"),
        dict(x0=0.0, label="local max", name="local max", color="C0"),
        dict(x0=3.5, label=None, name="local max", color="C0"),
    ]

    for lifter in [lifter4, lifter6]:
        thetas = np.linspace(*lifter.xlims, 100)
        costs = []
        for theta in thetas:
            cost = lifter.get_cost(theta)
            costs.append(cost)

        # local solver
        solution_list = lifter.solution_list

        x_global = None
        for dict_ in solution_list:
            if "max" in dict_["name"]:
                sol = minimize(lambda x: -lifter.get_cost(x), x0=dict_["x0"])
            else:
                sol = minimize(lifter.get_cost, x0=dict_["x0"])
            dict_["xhat"] = sol.x[0]
            if dict_["name"] == "global min":
                x_global = sol.x[0]

        fig, ax = plot_cost(thetas, costs)
        for i, dict_ in enumerate(solution_list):
            # cost_init = lifter.get_cost(dict_["x0"])
            # ax.scatter([dict_["x0"]], [cost_init], color=dict_["color"], marker="o")
            cost_hat = lifter.get_cost(dict_["xhat"])
            ax.scatter(
                [dict_["xhat"]],
                [cost_hat],
                color=dict_["color"],
                marker="x",
                label=dict_["label"],
            )
        ax.legend(loc="upper center")
        if SAVEFIG:
            savefig(fig, f"_plots/poly{lifter.degree}_local.png")

        # without redundant constraints
        Q = lifter.get_Q()
        A_known_wo_redundant = lifter.get_A_known(add_redundant=False)
        Constraints = lifter.get_A_b_list(A_known_wo_redundant)

        X, info_sdp = solve_sdp(Q, Constraints)
        x_hat, info_round = rank_project(X, p=1)
        t_hat = x_hat[1, 0]
        t_pick = X[1, 0]

        fig, ax = plot_cost(thetas, costs)
        plot_global(lifter, ax, {"global round": t_hat, "global pick": t_pick})
        if SAVEFIG: 
            savefig(fig, f"_plots/poly{lifter.degree}_global_wo.png")

        # with redundant constraints
        if lifter.degree > 4:
            A_known_w_redundant = lifter.get_A_known(add_redundant=True)
            Constraints = lifter.get_A_b_list(A_known_w_redundant)

            X, info_sdp = solve_sdp(Q, Constraints)
            x_hat, info_round = rank_project(X, p=1)
            t_hat = x_hat[1, 0]
            t_pick = X[1, 0]

            fig, ax = plot_cost(thetas, costs)
            plot_global(lifter, ax, {"global round": t_hat, "global pick": t_pick})
            if SAVEFIG:
                savefig(fig, f"_plots/poly{lifter.degree}_global.png")

    if not SAVEFIG:
        plt.show(block=True)
