                     
                       

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


METHODS = [
    {
        "id": "pure_sync",
        "display_name": "Pure-Sync",
        "aliases": [],
        "server_mode": "sync",
        "selection": "random",
        "mmqs": False,
        "prefetch": False,
        "mmqs_weight_mode": "static",
    },
    {
        "id": "pure_async",
        "display_name": "Pure-Async",
        "aliases": [],
        "server_mode": "async",
        "selection": "random",
        "mmqs": False,
        "prefetch": False,
        "mmqs_weight_mode": "static",
    },
    {
        "id": "hybrid",
        "display_name": "Hybrid",
        "aliases": [],
        "server_mode": "hybrid",
        "selection": "random",
        "mmqs": False,
        "prefetch": False,
        "mmqs_weight_mode": "static",
    },
    {
        "id": "mmqs_fw",
        "display_name": "MMQS-FW",
        "aliases": [],
        "server_mode": "hybrid",
        "selection": "mmqs",
        "mmqs": True,
        "prefetch": True,
        "mmqs_weight_mode": "static",
    },
    {
        "id": "mmqs_wo_mc",
        "display_name": "MMQS w/o MC",
        "aliases": [],
        "server_mode": "hybrid",
        "selection": "mmqs",
        "mmqs": True,
        "prefetch": True,
        "mmqs_weight_mode": "tot_api",
        "env_overrides": {
            "MMQS_TOT_DISABLE_MODAL": "1",
            "MMQS_TOT_MODAL": "0.0",
        },
    },
    {
        "id": "mmqs",
        "display_name": "MMQS",
        "aliases": [],
        "server_mode": "hybrid",
        "selection": "mmqs",
        "mmqs": True,
        "prefetch": True,
        "mmqs_weight_mode": "tot_api",
    },
]


def _normalize_method_token(raw):
    token = str(raw).strip().lower()
    for ch in (" ", "-", "_", "/"):
        token = token.replace(ch, "")
    return token


def _build_method_alias_map():
    alias_map = {}
    for method in METHODS:
        for token in [method["id"], method["display_name"]] + list(method.get("aliases", [])):
            alias_map[_normalize_method_token(token)] = method["id"]
    return alias_map


def parse_int_list(raw):
    vals = []
    for t in str(raw).split(","):
        t = t.strip()
        if t:
            vals.append(int(t))
    if not vals:
        raise ValueError("trials is empty")
    return vals


def parse_method_list(raw):
    alias_map = _build_method_alias_map()
    ids_all = [m["id"] for m in METHODS]
    vals = []
    for t in str(raw).split(","):
        t = t.strip()
        if t:
            vals.append(t)
    if not vals:
        raise ValueError("methods is empty")
    bad = [x for x in vals if _normalize_method_token(x) not in alias_map]
    if bad:
        raise ValueError("unknown methods: {} (available: {})".format(",".join(bad), ",".join(ids_all)))
    seen = set()
    uniq = []
    for x in vals:
        method_id = alias_map[_normalize_method_token(x)]
        if method_id not in seen:
            uniq.append(method_id)
            seen.add(method_id)
    return uniq


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_previous_success_keys(summary_path):
    keys = set()
    if not Path(summary_path).exists():
        return keys
    try:
        payload = read_json(summary_path)
    except Exception:
        return keys
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return keys
    for item in runs:
        if not isinstance(item, dict):
            continue
        try:
            method = str(item.get("method", "")).strip()
            trial = int(item.get("trial"))
        except Exception:
            continue
        if method == "":
            continue
        exit_code = int(item.get("exit_code", 1))
        timeout_hit = bool(item.get("timeout_hit", False))
        if exit_code == 0 and not timeout_hit:
            keys.add((method, trial))
    return keys


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


def build_cmd(args, cfg_path, method, trial):
    cmd = [
        args.python_exe,
        "run.py",
        "-c",
        str(cfg_path),
        "-sel",
        method["selection"],
        "--association",
        args.association,
        "--delay_mode",
        args.delay_mode,
        "--trial",
        str(int(trial)),
        "--log",
        args.log_level,
    ]
    if method["mmqs"]:
        cmd.extend(
            [
                "--mmqs_enabled",
                "--mmqs_weight_mode",
                str(method.get("mmqs_weight_mode", "static")),
                "--mmqs_loss_aware_topk_enabled",
                "--mmqs_W",
                "10",
                "--mmqs_beta",
                "5",
            ]
        )
    if method["prefetch"]:
        cmd.extend(
            [
                "--prefetch_enabled",
                "--prefetch_top_m",
                str(int(args.prefetch_top_m)),
                "--prefetch_delay_reduction_ratio",
                str(float(args.prefetch_delay_reduction_ratio)),
            ]
        )
    return cmd


def build_env_for_method(base_env, method):
    env_run = dict(base_env)
    for k, v in method.get("env_overrides", {}).items():
        env_run[str(k)] = str(v)
    return env_run


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[4]
    default_run_dir = (
        repo_root
        / "result"
        / "terminal_only_unused"
    )

    parser = argparse.ArgumentParser(description="AVE main8 run script")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--base_config", type=str, default="configs/AVE/hybrid_noniid_ave_tuned.json")
    parser.add_argument("--run_dir", type=str, default=str(default_run_dir))
    parser.add_argument("--data_path", type=str, default="./data/ave_fed/noniid_a1p0_c36_pt_fast")
    parser.add_argument(
        "--methods",
        type=str,
        default="pure_sync,pure_async,hybrid,mmqs_fw,mmqs_wo_mc,mmqs",
    )
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
    parser.add_argument(
        "--skip_existing_metrics",
        type=int,
        default=1,
        help="Skip method/trial when metrics csv already exists and non-empty (1/0).",
    )
    parser.add_argument(
        "--run_timeout_sec",
        type=int,
        default=7200,
        help="Timeout in seconds for each run.py call; <=0 disables timeout.",
    )
    parser.add_argument(
        "--continue_on_error",
        type=int,
        default=0,
        help="When 1, continue remaining runs even if one run fails.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = (repo_root / base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError("BASE_CONFIG_NOT_FOUND: {}".format(base_config_path))

    trials = parse_int_list(args.trials)
    methods_selected = parse_method_list(args.methods)
    methods_run = [m for m in METHODS if m["id"] in methods_selected]
    base_cfg = read_json(base_config_path)

    env = os.environ.copy()
    env["ALL_PROXY"] = ""
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["MMQS_SIM_COMM_PROFILE"] = "legacy_ave"
    terminal_only = True
    env["MMQS_TERMINAL_ONLY"] = "1"
    args.skip_existing_metrics = 0
    if not terminal_only:
        run_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    failed_runs = []
    run_count = 0
    summary_path = None if terminal_only else (run_dir / "run_summary.json")
    previous_success_keys = set() if terminal_only else load_previous_success_keys(summary_path)
    print("RUN_START=1")
    print("REPO_ROOT={}".format(repo_root))
    print("RUN_DIR={}".format(run_dir))
    print("METHODS={}".format([m["display_name"] for m in methods_run]))
    print("METHOD_IDS={}".format([m["id"] for m in methods_run]))
    print("TRIALS={}".format(trials))
    print("TERMINAL_ONLY={}".format(int(terminal_only)))
    print("PREV_SUCCESS_COUNT={}".format(len(previous_success_keys)))

    for method in methods_run:
        if terminal_only:
            method_dir = None
            method_cfg_dir = None
        else:
            method_dir = run_dir / str(method["id"])
            method_cfg_dir = method_dir / "configs"
            method_dir.mkdir(parents=True, exist_ok=True)
            method_cfg_dir.mkdir(parents=True, exist_ok=True)

        for trial in trials:
            run_count += 1
            cfg = json.loads(json.dumps(base_cfg))
            cfg.setdefault("paths", {})
            cfg.setdefault("server", {})
            cfg.setdefault("gateways", {})
            cfg.setdefault("federated_learning", {})
            cfg.setdefault("clients", {})

            cfg["paths"]["data"] = str(args.data_path)
            cfg["server"]["mode"] = str(method["server_mode"])
            method_rounds = int(args.sync_rounds) if str(method["server_mode"]) == "sync" else int(args.rounds)
            cfg["server"]["rounds"] = int(method_rounds)
            cfg["server"]["adjust_round"] = int(args.adjust_round)
            cfg["gateways"]["rounds"] = int(args.gateway_rounds)
            cfg["federated_learning"]["epochs"] = int(args.epochs)
            cfg["federated_learning"]["batch_size"] = int(args.batch_size)
            cfg["federated_learning"]["target_accuracy"] = float(args.target_accuracy)
            cfg["clients"]["total"] = int(args.clients_total)

            stem = "ave_main8_{}_t{}".format(method["id"], int(trial))
            if terminal_only:
                fd, tmp_cfg = tempfile.mkstemp(
                    prefix="mmqs_{}_t{}_".format(method["id"], int(trial)),
                    suffix=".json"
                )
                os.close(fd)
                cfg_path = Path(tmp_cfg)
            else:
                cfg_path = method_cfg_dir / (stem + ".json")
            log_path = None if terminal_only else (method_dir / (stem + ".log"))
            err_path = None if terminal_only else (method_dir / (stem + ".err.log"))
            metrics_path = None if terminal_only else (method_dir / (stem + ".metrics.csv"))
            cfg_created = False
            try:
                write_json(cfg_path, cfg)
                cfg_created = True

                key = (method["id"], int(trial))
                metrics_ready = (
                    (metrics_path is not None) and
                    metrics_path.exists() and
                    metrics_path.stat().st_size > 0
                )
                if int(args.skip_existing_metrics) != 0:
                    if key in previous_success_keys and metrics_ready:
                        item = {
                            "method": method["id"],
                            "method_display": method["display_name"],
                            "trial": int(trial),
                            "exit_code": 0,
                            "elapsed_sec": 0.0,
                            "config": str(cfg_path),
                            "log": None if log_path is None else str(log_path),
                            "err": None if err_path is None else str(err_path),
                            "metrics_csv": str(metrics_path),
                            "skipped_existing_metrics": True,
                            "timeout_hit": False,
                        }
                        summary.append(item)
                        print(
                            "RUN_SKIP METHOD={} TRIAL={} REASON=prev_success PATH={}".format(
                                method["display_name"], trial, metrics_path
                            )
                        )
                        continue
                    if metrics_ready and key not in previous_success_keys:
                        print(
                            "RUN_RETRY METHOD={} TRIAL={} REASON=metrics_exists_but_not_prev_success".format(
                                method["display_name"], trial
                            )
                        )
                before = {} if terminal_only else snapshot_ave_csv(repo_root)
                cmd = build_cmd(args, cfg_path, method, trial)
                print("RUN #{} METHOD={}({}) TRIAL={} ROUNDS={}".format(run_count, method["display_name"], method["id"], trial, int(method_rounds)))
                print("CMD={}".format(" ".join(cmd)))
                t0 = time.time()
                timeout_hit = False
                exit_code = 0
                timeout = int(args.run_timeout_sec)
                if timeout <= 0:
                    timeout = None
                env_run = build_env_for_method(env, method)
                if terminal_only:
                    try:
                        proc = subprocess.run(
                            cmd, cwd=str(repo_root), env=env_run, check=False, timeout=timeout
                        )
                        exit_code = int(proc.returncode)
                    except subprocess.TimeoutExpired:
                        timeout_hit = True
                        exit_code = 124
                        print(
                            "RUN_TIMEOUT METHOD={} TRIAL={} LIMIT_SEC={}".format(
                                method["display_name"], trial, int(args.run_timeout_sec)
                            )
                        )
                else:
                    with open(log_path, "w", encoding="utf-8", errors="ignore") as f_out, open(
                        err_path, "w", encoding="utf-8", errors="ignore"
                    ) as f_err:
                        try:
                            proc = subprocess.run(
                                cmd, cwd=str(repo_root), env=env_run, stdout=f_out, stderr=f_err, check=False, timeout=timeout
                            )
                            exit_code = int(proc.returncode)
                        except subprocess.TimeoutExpired:
                            timeout_hit = True
                            exit_code = 124
                            f_err.write(
                                "\n[RUN_TIMEOUT] method={} trial={} timeout_sec={}\n".format(
                                    method["id"], trial, int(args.run_timeout_sec)
                                )
                            )
                            f_err.flush()
                            print(
                                "RUN_TIMEOUT METHOD={} TRIAL={} LIMIT_SEC={}".format(
                                    method["display_name"], trial, int(args.run_timeout_sec)
                                )
                            )
                elapsed = time.time() - t0

                copied = None
                if not terminal_only:
                    updated = locate_updated_ave_csv(before, repo_root)
                    if updated is not None and Path(updated).exists():
                        shutil.copy2(str(updated), str(metrics_path))
                        copied = str(metrics_path)

                item = {
                    "method": method["id"],
                    "method_display": method["display_name"],
                    "trial": int(trial),
                    "rounds": int(method_rounds),
                    "exit_code": int(exit_code),
                    "elapsed_sec": float(elapsed),
                    "config": str(cfg_path),
                    "log": None if log_path is None else str(log_path),
                    "err": None if err_path is None else str(err_path),
                    "metrics_csv": copied,
                    "skipped_existing_metrics": False,
                    "timeout_hit": bool(timeout_hit),
                }
                summary.append(item)
                print(
                    "RUN_DONE METHOD={} TRIAL={} EXIT={} ELAPSED={:.2f}s TIMEOUT={}".format(
                        method["display_name"], trial, int(exit_code), elapsed, int(timeout_hit)
                    )
                )

                if exit_code != 0:
                    if not terminal_only:
                        write_json(summary_path, {"runs": summary})
                    failed_runs.append({
                        "method": method["id"],
                        "method_display": method["display_name"],
                        "trial": int(trial),
                        "err": "TERMINAL_STDERR" if err_path is None else str(err_path),
                        "exit_code": int(exit_code)
                    })
                    if int(args.continue_on_error) != 0:
                        print(
                            "RUN_FAIL_CONTINUE METHOD={} TRIAL={} EXIT={} ERR={}".format(
                                method["display_name"], trial, int(exit_code),
                                "TERMINAL_STDERR" if err_path is None else err_path
                            )
                        )
                        continue
                    raise RuntimeError(
                        "RUN_FAILED method={} trial={} err={}".format(
                            method["id"], trial,
                            "TERMINAL_STDERR" if err_path is None else err_path
                        )
                    )
            finally:
                if terminal_only and cfg_created:
                    try:
                        if cfg_path.exists():
                            cfg_path.unlink()
                    except OSError:
                        pass

    if not terminal_only:
        write_json(summary_path, {"runs": summary, "failed_runs": failed_runs})
        print("SUMMARY_JSON={}".format(summary_path))
    else:
        pass
    print("TOTAL_RUNS={}".format(run_count))
    print("FAILED_RUNS={}".format(len(failed_runs)))
    print("RUN_DONE=1")


if __name__ == "__main__":
    main()




