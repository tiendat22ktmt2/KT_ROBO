import numpy as np

from solvers.analytical import solve_analytical
from solvers.pinv import solve_pinv
from solvers.dls import solve_dls
from solvers.adaptive_dls import solve_adaptive_dls
from solvers.lm import solve_lm


SOLVER_FUNCTIONS = {

    "analytical": solve_analytical,

    "pinv": solve_pinv,

    "dls": solve_dls,

    "adaptive_dls": solve_adaptive_dls,

    "lm": solve_lm

}


def run_method(
    robot,
    targets,
    method,
    cfg,
    q_initial=None,
    elbow=1
):

    q = np.asarray(
        q_initial
        if q_initial is not None
        else [0.1, -0.2, 0.1],
        dtype=float
    )

    results = []

    for target in targets:

        if method == "analytical":

            result = solve_analytical(
                robot,
                target,
                elbow=elbow,
                q_seed=q,
                cfg=cfg
            )

        else:

            solver = SOLVER_FUNCTIONS[method]

            result = solver(
                robot,
                target,
                q,
                cfg
            )

        if result.converged:

            q = result.q.copy()

        results.append(result)

    return results