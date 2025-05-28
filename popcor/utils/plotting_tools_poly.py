import matplotlib.pyplot as plt


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
