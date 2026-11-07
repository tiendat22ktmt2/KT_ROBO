import matplotlib.pyplot as plt


def plot_singularity(
    t,
    sigma_min,
    path,
    title
):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        t,
        sigma_min
    )

    ax.set_xlabel("Time (s)")

    ax.set_ylabel(
        "Minimum singular value"
    )

    ax.set_title(title)

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)