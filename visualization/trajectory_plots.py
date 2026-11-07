import matplotlib.pyplot as plt


def plot_tracking(
    targets,
    achieved,
    path,
    title
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        targets[:, 0],
        targets[:, 1],
        label="Target",
        linewidth=2
    )

    ax.plot(
        achieved[:, 0],
        achieved[:, 1],
        "--",
        label="Achieved"
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    ax.set_title(title)

    ax.axis("equal")

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)