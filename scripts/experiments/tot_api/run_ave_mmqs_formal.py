                     
                       

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    default_config = repo_root / "configs" / "AVE" / "hybrid_noniid_ave_tuned.json"

    parser = argparse.ArgumentParser(description="AVE formal run: MMQS")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--log_path", type=str, default=str(here / "ave_tot_api_formal_t0.log"))
    parser.add_argument(
        "--api_url",
        type=str,
        default="",
        help="ToT API endpoint. Set this manually before running; no default credentials are provided.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="ToT API model name. Set this manually before running.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="",
        help="Environment variable containing the ToT API key. Set this manually before running.",
    )
    parser.add_argument("--proxy", type=str, default="")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--Q", type=int, default=5)
    parser.add_argument("--call_interval", type=int, default=3)
    parser.add_argument("--prune_interval", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError("CONFIG_NOT_FOUND: {}".format(cfg_path))

    terminal_only = True
    log_path = None
    if not terminal_only:
        log_path = Path(args.log_path)
        if not log_path.is_absolute():
            log_path = (repo_root / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if len(str(args.api_url).strip()) <= 0:
        raise RuntimeError("API_URL_EMPTY: pass your own --api_url")
    if len(str(args.model).strip()) <= 0:
        raise RuntimeError("MODEL_EMPTY: pass your own --model")
    if len(str(args.api_key_env).strip()) <= 0:
        raise RuntimeError("API_KEY_ENV_EMPTY: pass your own --api_key_env")

    env = os.environ.copy()
    if len(str(args.proxy).strip()) > 0:
        env["MMQS_TOT_API_PROXY"] = str(args.proxy)
        env["HTTP_PROXY"] = str(args.proxy)
        env["HTTPS_PROXY"] = str(args.proxy)
    env["MMQS_TOT_API_SINGLE_USER_ROLE"] = "1"
    env["MMQS_W_DATA"] = "0.2975"
    env["MMQS_W_PERF"] = "0.2975"
    env["MMQS_W_RES"] = "0.0850"
    env["MMQS_W_COOL"] = "0.0850"
    env["MMQS_W_FAIR"] = "0.0850"
    env["MMQS_W_MODAL"] = "0.1500"
    env["MMQS_SIM_COMM_PROFILE"] = "legacy_ave"
    env["MMQS_TERMINAL_ONLY"] = "1"
    cmd = [
        str(args.python_exe),
        "run.py",
        "-c",
        str(cfg_path),
        "-sel",
        "mmqs",
        "--association",
        "random",
        "--delay_mode",
        "nycmesh",
        "--trial",
        str(int(args.trial)),
        "--log",
        "INFO",
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
        "4",
        "--prefetch_delay_reduction_ratio",
        "0.25",
        "--mmqs_tot_api_enabled",
        "--mmqs_tot_api_url",
        str(args.api_url),
        "--mmqs_tot_api_model",
        str(args.model),
        "--mmqs_tot_api_key_env",
        str(args.api_key_env),
        "--mmqs_tot_api_timeout",
        str(float(args.timeout)),
        "--mmqs_tot_api_max_tokens",
        str(int(args.max_tokens)),
        "--mmqs_tot_api_temperature",
        str(float(args.temperature)),
        "--mmqs_tot_api_top_p",
        str(float(args.top_p)),
        "--mmqs_tot_api_retry",
        str(int(args.retry)),
        "--mmqs_tot_q",
        str(int(args.Q)),
        "--mmqs_tot_api_call_interval",
        str(int(args.call_interval)),
        "--mmqs_tot_api_prune_interval",
        str(int(args.prune_interval)),
        "--mmqs_tot_api_force_non_stream",
    ]

    print("RUN_SCRIPT=run_ave_mmqs_formal")
    print("REPO_ROOT={}".format(repo_root))
    print("CONFIG={}".format(cfg_path))
    print("TERMINAL_ONLY={}".format(int(terminal_only)))
    print("LOG={}".format("DISABLED_TERMINAL_ONLY" if log_path is None else log_path))
    print("CMD={}".format(subprocess.list2cmdline(cmd)))

    if terminal_only:
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    else:
        with open(log_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("RUN_SCRIPT=run_ave_mmqs_formal\n")
            f.write("REPO_ROOT={}\n".format(repo_root))
            f.write("CONFIG={}\n".format(cfg_path))
            f.write("CMD={}\n".format(subprocess.list2cmdline(cmd)))
            f.flush()
            proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT, check=False)

    print("EXIT_CODE={}".format(int(proc.returncode)))
    print("LOG_PATH={}".format("DISABLED_TERMINAL_ONLY" if log_path is None else log_path))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
