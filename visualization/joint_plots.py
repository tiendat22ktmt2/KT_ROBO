import matplotlib.pyplot as plt
import numpy as np


def plot_joint_profiles(
    t,
    q,
    path,
    title
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for i in range(3):

        ax.plot(
            t,
            np.rad2deg(q[:, i]),
            label=f"q{i+1}"
        )

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("Joint angle (deg)")

    ax.set_title(title)

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)