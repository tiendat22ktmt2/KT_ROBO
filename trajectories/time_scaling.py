import numpy as np


def time_scaling(
    profile,
    samples,
    duration
):

    t = np.linspace(
        0.0,
        duration,
        samples
    )

    tau = t / duration

    profile = profile.lower()

    if profile == "cubic":

        s = (
            3 * tau**2
            - 2 * tau**3
        )

        ds = (
            6 * tau
            - 6 * tau**2
        ) / duration

        dds = (
            6 - 12 * tau
        ) / duration**2

    elif profile == "quintic":

        s = (
            10 * tau**3
            - 15 * tau**4
            + 6 * tau**5
        )

        ds = (
            30 * tau**2
            - 60 * tau**3
            + 30 * tau**4
        ) / duration

        dds = (
            60 * tau
            - 180 * tau**2
            + 120 * tau**3
        ) / duration**2

    elif profile == "lspb":

        tb = 0.2

        a = 1.0 / (
            tb * (1.0 - tb)
        )

        vmax = a * tb

        s = np.empty_like(tau)
        ds = np.empty_like(tau)
        dds = np.empty_like(tau)

        first = tau < tb

        middle = (
            (tau >= tb)
            & (tau <= 1.0 - tb)
        )

        last = tau > 1.0 - tb

        s[first] = (
            0.5 * a * tau[first]**2
        )

        ds[first] = (
            a * tau[first]
        )

        dds[first] = a

        s[middle] = (
            0.5 * a * tb**2
            + vmax * (
                tau[middle] - tb
            )
        )

        ds[middle] = vmax
        dds[middle] = 0.0

        u = 1.0 - tau[last]

        s[last] = (
            1.0 - 0.5 * a * u**2
        )

        ds[last] = a * u
        dds[last] = -a

        ds /= duration
        dds /= duration**2

    else:

        raise ValueError(
            f"Unknown time profile: {profile}"
        )

    return t, s, ds, dds


def apply_time_scaling(
    points,
    profile,
    duration
):

    points = np.asarray(
        points,
        dtype=float
    )

    t, s, ds, dds = time_scaling(
        profile,
        len(points),
        duration
    )

    u = np.linspace(
        0.0,
        1.0,
        len(points)
    )

    target = np.column_stack([

        np.interp(
            s,
            u,
            points[:, 0]
        ),

        np.interp(
            s,
            u,
            points[:, 1]
        )

    ])

    return t, target, ds, dds