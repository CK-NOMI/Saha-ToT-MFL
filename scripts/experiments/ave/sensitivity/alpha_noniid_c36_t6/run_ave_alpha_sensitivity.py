                     
                       

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_float_list(raw):
    vals = []
    for t in str(raw).split(","):
        t = t.strip()
        if t:
            vals.append(float(t))
    if not vals:
        raise ValueError("alphas is empty")
    return vals


def parse_int_list(raw):
    vals = []
    for t in str(raw).split(","):
        t = t.strip()
        if t:
            vals.append(int(t))
    if not vals:
        raise ValueError("trials is empty")
    return vals


def alpha_tag(alpha):
    return "{:.1f}".format(float(alpha)).replace(".", "p")


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def snapshot_ave_csv(repo_root):
    out = {}
    for p in repo_root.glob("AVE_*.csv"):
        try:
            st = p.stat()
            out[str(p.resolve())] = (st.st_mtime, st.st_size)
        except OSError:
            continue
    return out


def locate_updated_ave_csv(before, repo_root):
    cands = []
    for p in repo_root.glob("AVE_*.csv"):
        try:
            st = p.stat()
        except OSError:
            continue
        key = str(p.resolve())
        old = before.get(key)
        if old is None or st.st_mtime > old[0] + 1e-9 or st.st_size != old[1]:
            cands.append((st.st_mtime, p.resolve()))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def build_cmd(args, cfg_path, trial):
    return [
        args.python_exe,
        "run.py",
        "-c",
        str(cfg_path),
        "-sel",
        "mmqs",
        "--mmqs_enabled",
        "--mmqs_weight_mode",
        "tot_api",
        "--mmqs_loss_aware_topk_enabled",
        "--mmqs_W",
        "10",
        "--mmqs_beta",
        "5",
        "--prefetch_enabled",
        "--prefetch_top_m",
        str(int(args.prefetch_top_m)),
        "--prefetch_delay_reduction_ratio",
        str(float(args.prefetch_delay_reduction_ratio)),
        "--association",
        args.association,
        "--delay_mode",
        args.delay_mode,
        "--trial",
        str(int(trial)),
        "--log",
        args.log_level,
    ]


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[4]
    default_run_dir = repo_root / "result" / "experiments" / "ave_results" / "exp3" / "ave_alpha_sensitivity_noniid_c36_r850_t6"

    parser = argparse.ArgumentParser(description="AVE Dirichlet alpha sensitivity runner (main method: MMQS+Prefetch+ToT-Lite)")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--base_config", type=str, default="configs/AVE/hybrid_noniid_ave_tuned.json")
    parser.add_argument("--run_dir", type=str, default=str(default_run_dir))
    parser.add_argument("--data_root", type=str, default="./data/ave_fed")
    parser.add_argument("--alphas", type=str, default="0.1,0.3,0.5,1.0")
    parser.add_argument("--trials", type=str, default="0,1,2,3,4")
    parser.add_argument("--rounds", type=int, default=850)
    parser.add_argument("--adjust_round", type=int, default=20)
    parser.add_argument("--gateway_rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_accuracy", type=float, default=0.0)
    parser.add_argument("--association", type=str, default="random")
    parser.add_argument("--delay_mode", type=str, default="nycmesh")
    parser.add_argument("--prefetch_top_m", type=int, default=4)
    parser.add_argument("--prefetch_delay_reduction_ratio", type=float, default=0.25)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_dir = run_dir / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = (repo_root / base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError("BASE_CONFIG_NOT_FOUND: {}".format(base_config_path))

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()

    base_cfg = read_json(base_config_path)
    alphas = parse_float_list(args.alphas)
    trials = parse_int_list(args.trials)

    env = os.environ.copy()
    env["ALL_PROXY"] = ""
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["MMQS_SIM_COMM_PROFILE"] = "legacy_ave"
    terminal_only = True
    env["MMQS_TERMINAL_ONLY"] = "1"

    summary = []
    run_count = 0
    print("RUN_START=1")
    print("ALPHAS={}".format(alphas))
    print("TRIALS={}".format(trials))
    print("TERMINAL_ONLY={}".format(int(terminal_only)))

    for alpha in alphas:
        tag = alpha_tag(alpha)
        data_dir = data_root / ("noniid_a{}_c36_pt_fast".format(tag))
        if not data_dir.exists():
            raise FileNotFoundError("DATA_PARTITION_NOT_FOUND alpha={} path={}".format(alpha, data_dir))

        for trial in trials:
            run_count += 1
            cfg = json.loads(json.dumps(base_cfg))
            cfg.setdefault("paths", {})
            cfg.setdefault("server", {})
            cfg.setdefault("gateways", {})
            cfg.setdefault("federated_learning", {})

            cfg["paths"]["data"] = str(data_dir).replace(str(repo_root) + os.sep, "./").replace("\\", "/")
            cfg["server"]["mode"] = "hybrid"
            cfg["server"]["rounds"] = int(args.rounds)
            cfg["server"]["adjust_round"] = int(args.adjust_round)
            cfg["gateways"]["rounds"] = int(args.gateway_rounds)
            cfg["federated_learning"]["epochs"] = int(args.epochs)
            cfg["federated_learning"]["batch_size"] = int(args.batch_size)
            cfg["federated_learning"]["target_accuracy"] = float(args.target_accuracy)

            stem = "ave_alpha_tot_{}_t{}".format(tag, int(trial))
            cfg_path = cfg_dir / (stem + (".cfg" if terminal_only else ".json"))
            log_path = None if terminal_only else (run_dir / (stem + ".log"))
            err_path = None if terminal_only else (run_dir / (stem + ".err.log"))
            metrics_path = run_dir / (stem + ".metrics.csv")
            write_json(cfg_path, cfg)

            before = snapshot_ave_csv(repo_root)
            cmd = build_cmd(args, cfg_path, trial)
            print("RUN #{} ALPHA={} TRIAL={}".format(run_count, alpha, trial))
            print("CMD={}".format(" ".join(cmd)))
            t0 = time.time()
            if terminal_only:
                proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
            else:
                with open(log_path, "w", encoding="utf-8", errors="ignore") as f_out, open(
                    err_path, "w", encoding="utf-8", errors="ignore"
                ) as f_err:
                    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=f_out, stderr=f_err, check=False)
            elapsed = time.time() - t0

            copied = None
            if not terminal_only:
                updated = locate_updated_ave_csv(before, repo_root)
                if updated is not None and Path(updated).exists():
                    shutil.copy2(str(updated), str(metrics_path))
                    copied = str(metrics_path)
            if terminal_only:
                for p in (cfg_path, metrics_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass

            item = {
                "alpha": float(alpha),
                "trial": int(trial),
                "exit_code": int(proc.returncode),
                "elapsed_sec": float(elapsed),
                "config": str(cfg_path),
                "log": None if log_path is None else str(log_path),
                "err": None if err_path is None else str(err_path),
                "metrics_csv": copied,
            }
            summary.append(item)
            print("RUN_DONE ALPHA={} TRIAL={} EXIT={} ELAPSED={:.2f}s".format(alpha, trial, int(proc.returncode), elapsed))
            if proc.returncode != 0:
                if not terminal_only:
                    write_json(run_dir / "run_summary.json", {"runs": summary})
                raise RuntimeError("RUN_FAILED alpha={} trial={} err={}".format(
                    alpha, trial, "TERMINAL_STDERR" if err_path is None else err_path
                ))

    if not terminal_only:
        write_json(run_dir / "run_summary.json", {"runs": summary})
    else:
        try:
            p = run_dir / "run_summary.json"
            if p.exists():
                p.unlink()
        except OSError:
            pass
    print("TOTAL_RUNS={}".format(run_count))
    print("RUN_DONE=1")


if __name__ == "__main__":
    main()


