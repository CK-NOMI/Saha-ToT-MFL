                     
                       

import argparse
import os
import subprocess
import sys
from pathlib import Path


EXP_MATRIX = [
    {
        "name": "mmqs_fw_full",
        "method": "mmqs_fw",
        "static_weights": [0.2975, 0.2975, 0.0850, 0.0850, 0.0850, 0.1500],
        "tot_mode": "full",
    },
    {
        "name": "mmqs_fw_wo_mc",
        "method": "mmqs_fw",
        "static_weights": [0.3500, 0.3500, 0.1000, 0.1000, 0.1000, 0.0000],
        "tot_mode": "off",
    },
    {
        "name": "mmqs_full",
        "method": "mmqs",
        "static_weights": [0.2975, 0.2975, 0.0850, 0.0850, 0.0850, 0.1500],
        "tot_mode": "full",
    },
    {
        "name": "mmqs_wo_mc",
        "method": "mmqs_wo_mc",
        "static_weights": [0.3500, 0.3500, 0.1000, 0.1000, 0.1000, 0.0000],
        "tot_mode": "off",
    },
]


def _run_dir_name(exp_name, rounds, trials, tag):
    trial_token = str(trials).replace(",", "_").replace(" ", "")
    base = "ave6d_{}_noniid_a1p0_c36_r{}_t{}".format(exp_name, int(rounds), trial_token)
    if str(tag).strip():
        base += "_" + str(tag).strip()
    return base


def _build_env(base_env, static_weights, tot_mode):
    env = dict(base_env)
    env["MMQS_SIM_COMM_PROFILE"] = "legacy_ave"
    keys = [
        "MMQS_W_DATA",
        "MMQS_W_PERF",
        "MMQS_W_RES",
        "MMQS_W_COOL",
        "MMQS_W_FAIR",
        "MMQS_W_MODAL",
    ]
    for key, value in zip(keys, static_weights):
        env[key] = "{:.6f}".format(float(value))

    if str(tot_mode).lower() == "off":
        env["MMQS_TOT_DISABLE_MODAL"] = "1"
        env["MMQS_TOT_MODAL"] = "0.0"
    else:
        env["MMQS_TOT_DISABLE_MODAL"] = "0"
        env["MMQS_TOT_MODAL"] = "0.15"
    return env


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[4]
    default_runner = repo_root / "scripts" / "experiments" / "ave" / "main8" / "noniid_a1p0_c36_t6" / "run_ave_main8.py"

    parser = argparse.ArgumentParser(description="AVE 6-dimension scoring (modal ablation) 4-group runner")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--runner", type=str, default=str(default_runner))
    parser.add_argument("--run_root", type=str, default=str(here))
    parser.add_argument("--base_config", type=str, default="configs/AVE/hybrid_noniid_ave_tuned.json")
    parser.add_argument("--data_path", type=str, default="./data/ave_fed/noniid_a1p0_c36_pt_fast")
    parser.add_argument("--trials", type=str, default="0,1,2,3,4")
    parser.add_argument("--rounds", type=int, default=850)
    parser.add_argument("--sync_rounds", type=int, default=150)
    parser.add_argument("--adjust_round", type=int, default=20)
    parser.add_argument("--gateway_rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_accuracy", type=float, default=0.0)
    parser.add_argument("--clients_total", type=int, default=36)
    parser.add_argument("--association", type=str, default="random")
    parser.add_argument("--delay_mode", type=str, default="nycmesh")
    parser.add_argument("--prefetch_top_m", type=int, default=4)
    parser.add_argument("--prefetch_delay_reduction_ratio", type=float, default=0.25)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--skip_existing_metrics", type=int, default=1)
    parser.add_argument("--run_timeout_sec", type=int, default=30000)
    parser.add_argument("--continue_on_error", type=int, default=1)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = (repo_root / runner).resolve()
    if not runner.exists():
        raise FileNotFoundError("RUNNER_NOT_FOUND: {}".format(runner))

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    print("RUN_SCRIPT=run_ave_6d_modal_main8")
    print("REPO_ROOT={}".format(repo_root))
    print("RUNNER={}".format(runner))
    print("RUN_ROOT={}".format(run_root))
    print("TRIALS={}".format(args.trials))
    print("ROUNDS={}".format(args.rounds))

    any_fail = False
    for idx, exp in enumerate(EXP_MATRIX, start=1):
        run_dir = run_root / _run_dir_name(exp["name"], args.rounds, args.trials, args.tag)
        env = _build_env(os.environ, exp["static_weights"], exp["tot_mode"])
        cmd = [
            args.python_exe,
            str(runner),
            "--python_exe", args.python_exe,
            "--base_config", str(args.base_config),
            "--run_dir", str(run_dir),
            "--data_path", str(args.data_path),
            "--methods", str(exp["method"]),
            "--trials", str(args.trials),
            "--rounds", str(int(args.rounds)),
            "--sync_rounds", str(int(args.sync_rounds)),
            "--adjust_round", str(int(args.adjust_round)),
            "--gateway_rounds", str(int(args.gateway_rounds)),
            "--epochs", str(int(args.epochs)),
            "--batch_size", str(int(args.batch_size)),
            "--target_accuracy", str(float(args.target_accuracy)),
            "--clients_total", str(int(args.clients_total)),
            "--association", str(args.association),
            "--delay_mode", str(args.delay_mode),
            "--prefetch_top_m", str(int(args.prefetch_top_m)),
            "--prefetch_delay_reduction_ratio", str(float(args.prefetch_delay_reduction_ratio)),
            "--log_level", str(args.log_level),
            "--skip_existing_metrics", str(int(args.skip_existing_metrics)),
            "--run_timeout_sec", str(int(args.run_timeout_sec)),
            "--continue_on_error", str(int(args.continue_on_error)),
        ]

        print("EXP #{} NAME={}".format(idx, exp["name"]))
        print("METHOD={}".format(exp["method"]))
        print("RUN_DIR={}".format(run_dir))
        print("STATIC_WEIGHTS={}".format(exp["static_weights"]))
        print("TOT_MODE={} TOT_MODAL={} TOT_DISABLE={}".format(
            exp["tot_mode"], env.get("MMQS_TOT_MODAL", ""), env.get("MMQS_TOT_DISABLE_MODAL", "")
        ))
        print("CMD={}".format(" ".join(cmd)))

        if int(args.dry_run) != 0:
            continue

        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
        print("EXIT_CODE={}".format(int(proc.returncode)))
        if int(proc.returncode) != 0:
            any_fail = True
            if int(args.continue_on_error) == 0:
                break

    if any_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

