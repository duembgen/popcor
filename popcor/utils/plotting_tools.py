import os
from collections.abc import Iterable

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix

from popcor.utils.geometry import get_C_r_from_theta


def import_plt():
    import shutil

    import matplotlib.pylab as plt

    usetex = True if shutil.which("latex") else False
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )
    plt.rc("text.latex", preamble=r"\usepackage{bm}")
    return plt


plt = import_plt()

FIGSIZE = 4


def add_colorbar(fig, ax, im, title=None, nticks=None, visible=True, size=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    if size is None:
        w, h = fig.get_size_inches()
        size = f"{5*h/w}%"
    cax = divider.append_axes("right", size=size, pad=0.05)
    if title is not None:
        cax.set_ylabel(title)

    if not visible:
        cax.axis("off")
        return cax

    fig.colorbar(im, cax=cax, orientation="vertical")

    # add symmetric nticks ticks: min and max, and equally spaced in between
    if nticks is not None:
        from math import floor

        ticks = cax.get_yticks()
        new_ticks = [ticks[0]]
        step = floor(len(ticks) / (nticks - 1))
        new_ticks += list(ticks[step + 1 :: step])
        new_ticks += [ticks[-1]]
        # print(f"reduce {ticks} to {new_ticks}")
        assert len(new_ticks) == nticks
        cax.set_yticks(ticks[::step])
    return cax


def add_scalebar(
    ax, size=5, size_vertical=1, loc="lower left", fontsize=8, color="black", pad=0.1
):
    """Add a scale bar to the plot.

    :param ax: axis to use.
    :param size: size of scale bar.
    :param size_vertical: height (thckness) of the bar
    :param loc: location (same syntax as for matplotlib legend)
    """
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax.transData,
        size,
        "{} m".format(size),
        loc,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=size_vertical,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)


def make_dirs_safe(path):
    """Make directory of input path, if it does not exist yet."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fig, name, verbose=True):
    make_dirs_safe(name)
    fig.savefig(name, bbox_inches="tight", pad_inches=0, dpi=200)
    if verbose:
        print(f"saved plot as {name}")


def plot_frame(
    ax,
    theta,
    color="k",
    marker="+",
    label=None,
    scale=1.0,
    ls="--",
    alpha=0.5,
    d=3,
    **kwargs,
):
    if np.ndim(theta) == 2:
        # used by rotation averaging
        C_cw = theta
        r_wc_w = np.zeros((theta.shape[0]))
    else:
        try:
            C_cw, r_wc_c = get_C_r_from_theta(theta, d=d)
            r_wc_w = -C_cw.T @ r_wc_c  # r_wc_w
        except Exception as e:
            C_cw = None
            r_wc_w = theta

    assert r_wc_w is not None

    if C_cw is not None:
        for col, dir_gt in zip(["r", "g", "b"], C_cw):
            length = scale / np.linalg.norm(dir_gt)
            ax.plot(
                [r_wc_w[0], r_wc_w[0] + length * dir_gt[0]],
                [r_wc_w[1], r_wc_w[1] + length * dir_gt[1]],
                color=col,
                ls=ls,
                alpha=alpha,
                zorder=-1,
            )
    ax.scatter(
        *r_wc_w[:2].T,
        color=color,
        marker=marker,
        label=None,
        zorder=1,
        **kwargs,
    )
    ax.plot(
        [],
        [],
        marker=marker,
        label=label,
        zorder=1,
        ls=ls,
        color="k",
        **kwargs,
    )
    return r_wc_w, C_cw


def add_rectangles(ax, dict_sizes, color="red"):
    from matplotlib.patches import Rectangle

    cumsize = 0
    xticks = []
    xticklabels = []
    for key, size in dict_sizes.items():
        cumsize += size
        xticks.append(cumsize - 0.5)
        xticklabels.append(f"${key}$")
        ax.add_patch(Rectangle((-0.5, -0.5), cumsize, cumsize, ec=color, fc="none"))
        # ax.annotate(text=key, xy=(cumsize, 1), color='red', weight="bold")
    ax.set_xticks(xticks, xticklabels)
    ax.tick_params(axis="x", colors="red")
    ax.xaxis.tick_top()
    ax.set_yticks([])


def initialize_discrete_cbar(values):
    import matplotlib.colors

    values = sorted(list(np.unique(values.round(3))) + [0])
    cmap = plt.get_cmap("viridis", len(values))
    cmap.set_over((1.0, 0.0, 0.0))
    cmap.set_under((0.0, 0.0, 1.0))
    bounds = [values[0] - 0.005] + [v + 0.005 for v in values]
    colorbar_yticks = [""] + list(values)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colorbar_yticks


def plot_basis(
    basis_poly: PolyMatrix,
    variables_j: dict,
    variables_i: Iterable | None = None,
    fname_root: str = "",
    discrete: bool = True,
):
    if variables_i is None:
        variables_i = basis_poly.generate_variable_dict_i()

    if discrete:
        values = basis_poly.get_matrix((variables_i, variables_j)).data  # type: ignore
        cmap, norm, colorbar_yticks = initialize_discrete_cbar(values)
    else:
        cmap = plt.get_cmap("viridis")
        norm = None
        colorbar_yticks = None

    # reduced_ticks below has no effect because all variables in variables_j are of size 1.
    fig, ax, im = basis_poly.matshow(
        variables_i=variables_i,
        variables_j=variables_j,
        cmap=cmap,
        norm=norm,  # reduced_ticks=True
    )
    fig.set_size_inches(15, 15 * len(variables_i) / len(variables_j))  # type: ignore
    cax = add_colorbar(fig, ax, im)
    if colorbar_yticks is not None:
        cax.set_yticklabels(colorbar_yticks)
    if fname_root != "":
        savefig(fig, fname_root + f"_basis.png")
    return fig, ax


def plot_tightness(df_tight, qcqp_cost, fname_root):
    fig, ax = plt.subplots()
    ax.axhline(qcqp_cost, color="k")
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        try:
            number = np.where(df_sub.tight.values)[0][0]
            label = f"{order_name} {order_type}: {number}"
        except IndexError:
            number = None
            label = f"{order_name} {order_type}: never tight"
        ax.semilogy(range(1, len(df_sub) + 1), np.abs(df_sub.cost.values), label=label)
    ax.legend()
    ax.grid()
    if fname_root != "":
        savefig(fig, fname_root + f"_tightness.png")


def plot_matrix(
    Ai,
    vmin=None,
    vmax=None,
    nticks=None,
    title="",
    colorbar=True,
    fig=None,
    ax=None,
    log=True,
    discrete=False,
):
    import matplotlib
    import matplotlib.colors

    if ax is None:
        fig, ax = plt.subplots()
    if fig is None:
        fig = plt.gcf()

    norm = None
    if log:
        norm = matplotlib.colors.SymLogNorm(10**-5, vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("seismic")  # midpoint is white
    colorbar_yticks = None
    if type(Ai) is np.ndarray:
        Ai_array = Ai
    else:
        Ai_array = Ai.toarray()

    if discrete:
        values = np.unique(Ai_array[Ai_array != 0])
        nticks = None
        cmap, norm, colorbar_yticks = initialize_discrete_cbar(values)

    im = ax.matshow(Ai_array, norm=norm, cmap=cmap)
    ax.axis("off")
    ax.set_title(title, y=1.0)
    if colorbar:
        cax = add_colorbar(fig, ax, im, nticks=nticks)
        if colorbar_yticks is not None:
            cax.set_yticklabels(colorbar_yticks)
    else:
        cax = add_colorbar(fig, ax, im, nticks=nticks, visible=False)
    return fig, ax, im


def plot_matrices(df_tight, fname_root):
    import itertools
    from math import ceil

    # for (order, order_type), df_sub in df_tight.groupby(["order", "type"]):
    matrix_types = ["A", "H"]
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        A_agg = None
        H = None

        n_cols = 10
        n_rows = min(ceil(len(df_sub) / n_cols), 5)  # plot maximum 5 * 2 rows
        fig, axs = plt.subplots(n_rows * 2, n_cols)
        fig.set_size_inches(n_cols, n_rows * 2)
        for j, matrix_type in enumerate(matrix_types):
            matrices = df_sub[matrix_type].values
            names_here = df_sub.name.values
            costs = df_sub.cost.values

            # make sure it's symmetric
            vmin = np.min([np.min(A) for A in matrices if (A is not None)])
            vmax = np.max([np.max(A) for A in matrices if (A is not None)])
            vmin = min(vmin, -vmax)
            vmax = max(vmax, -vmin)

            i = 0

            for row, col in itertools.product(range(n_rows), range(n_cols)):
                ax = axs[row * 2 + j, col]
                if i < len(matrices):
                    if matrices[i] is None:
                        continue
                    title = f"{matrix_type}{i}"
                    if matrix_type == "A":
                        title += f"\n{names_here[i]}"
                    else:
                        title += f"\nc={costs[i]:.2e}"

                    plot_matrix(
                        ax=ax,
                        Ai=matrices[i],
                        vmin=vmin,
                        vmax=vmax,
                        title=title,
                        colorbar=False,
                    )
                    ax.set_title(title, fontsize=5)

                    if matrix_type == "A":
                        if A_agg is None:
                            A_agg = np.abs(matrices[i].toarray()) > 1e-10
                        else:
                            A_agg = np.logical_or(
                                A_agg, (np.abs(matrices[i].toarray()) > 1e-10)
                            )
                    elif matrix_type == "H":
                        # make sure we store the last valid estimate of H, for plotting
                        if matrices[i] is not None:
                            H = matrices[i]
                else:
                    ax.axis("off")
                i += 1

        fname = fname_root + f"_{order_name}_{order_type}.png"
        savefig(fig, fname)
        return


def plot_singular_values(S, eps=None, label: str | None = "singular values", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    fig.set_size_inches(4, 2)
    ax.semilogy(S, marker="o", label=label)
    if eps is not None:
        ax.axhline(eps, color="C1")
    ax.grid()
    ax.set_xlabel("index")
    ax.set_ylabel("abs. singular values")
    if label is not None:
        ax.legend(loc="upper right")
    return fig, ax


def plot_aggregate_sparsity(mask):
    fig, ax = plt.subplots()
    ax.matshow(mask.toarray())
    plt.show(block=False)
    return fig, ax


def add_lines(ax, xs, start, facs=[1, 2, 3]):
    for fac in facs:
        ys = xs**fac / (np.min(xs) ** fac) * start
        ax.plot(
            xs,
            ys,
            color="k",
            alpha=0.5,
            ls=":",
        )
        ax.annotate(xy=(xs[-2], ys[-2] * 0.7), text=f"O(N$^{fac}$)", alpha=0.5)
