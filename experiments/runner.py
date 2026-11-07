from pathlib import Path

import numpy as np
import pandas as pd

from trajectories.scenarios import (
    make_scenario,
    make_pose_targets,
    target_feasibility
)

from trajectories.time_scaling import (
    apply_time_scaling
)

from .simulation import run_method

from evaluation.metrics import evaluate_run

from visualization.trajectory_plots import (
    plot_tracking
)

from visualization.joint_plots import (
    plot_joint_profiles
)

from visualization.singularity_plots import (
    plot_singularity
)


def run_all(
    robot,
    solver_cfg,
    exp_cfg
):

    output_dir = Path(
        exp_cfg.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    for scenario in exp_cfg.scenarios:

        xy, metadata = make_scenario(
            scenario,
            robot,
            exp_cfg.samples,
            exp_cfg.shape_scale,
            exp_cfg.shape_center,
            exp_cfg.orientation
        )

        for profile in exp_cfg.profiles:

            t, timed_xy, _, _ = apply_time_scaling(
                xy,
                profile,
                exp_cfg.duration
            )

            targets = make_pose_targets(
                timed_xy,
                exp_cfg.orientation
            )

            # Check each target independently.
            feasible = target_feasibility(
                robot,
                targets
            )

            for method in exp_cfg.methods:

                print(
                    f"\nRunning {scenario} | "
                    f"{profile} | {method}"
                )

                results = run_method(
                    robot,
                    targets,
                    method,
                    solver_cfg
                )

                q = np.vstack([
                    result.q
                    for result in results
                ])

                achieved = np.vstack([
                    robot.fk(qi)
                    for qi in q
                ])

                metrics, frame = evaluate_run(
                    t,
                    targets,
                    achieved,
                    q,
                    results,
                    feasible
                )

                metrics.update({

                    "scenario": scenario,

                    "profile": profile,

                    "method": method

                })

                case_dir = (
                    output_dir
                    / scenario
                    / profile
                    / method
                )

                case_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                frame.to_csv(
                    case_dir / "samples.csv",
                    index=False
                )

                plot_tracking(
                    targets,
                    achieved,
                    case_dir / "tracking.png",
                    f"{scenario} / {profile} / {method}"
                )

                plot_joint_profiles(
                    t,
                    q,
                    case_dir / "joint_profiles.png",
                    f"{scenario} / {profile} / {method}"
                )

                plot_singularity(
                    t,
                    [r.sigma_min for r in results],
                    case_dir / "singularity.png",
                    f"{scenario} / {profile} / {method}"
                )

                summary.append(metrics)

                print(
                    f"RMSE = "
                    f"{metrics['position_rmse_mm']:.6f} mm | "
                    f"Convergence = "
                    f"{metrics['convergence_rate_pct']:.2f}%"
                )

    summary_df = pd.DataFrame(summary)

    summary_dir = output_dir / "summary"

    summary_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        summary_dir / "all_metrics.csv",
        index=False
    )

    summary_df.to_excel(
        summary_dir / "all_metrics.xlsx",
        index=False
    )

    print("\n===================================")
    print("ALL EXPERIMENTS COMPLETED")
    print("===================================")

    columns = [
        "scenario",
        "profile",
        "method",
        "position_rmse_mm",
        "convergence_rate_pct"
    ]

    print(
        summary_df[columns].to_string(
            index=False
        )
    )

    return summary_df