                     
                       

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def format_float_tag(value):
    return ("{:.3f}".format(float(value))).rstrip("0").rstrip(".").replace(".", "p")


def alpha_subdir(alpha):
    return "noniid_a{}_c36_pt_fast".format("{:.1f}".format(float(alpha)).replace(".", "p"))


def build_tasks(missing_rate):
    miss_tag = format_float_tag(missing_rate)
    return [
        {
            "name": "alpha_a0p1",
            "output_subdir": alpha_subdir(0.1),
            "dirichlet_alpha": 0.1,
            "missing_image": 0.0,
            "missing_audio": 0.0,
            "missing_text": 0.0,
        },
        {
            "name": "alpha_a0p5",
            "output_subdir": alpha_subdir(0.5),
            "dirichlet_alpha": 0.5,
            "missing_image": 0.0,
            "missing_audio": 0.0,
            "missing_text": 0.0,
        },
        {
            "name": "alpha_a1p0",
            "output_subdir": alpha_subdir(1.0),
            "dirichlet_alpha": 1.0,
            "missing_image": 0.0,
            "missing_audio": 0.0,
            "missing_text": 0.0,
        },
        {
            "name": "miss_image",
            "output_subdir": "noniid_a1p0_c36_pt_fast_mi{}_ma0_mt0".format(miss_tag),
            "dirichlet_alpha": 1.0,
            "missing_image": float(missing_rate),
            "missing_audio": 0.0,
            "missing_text": 0.0,
        },
        {
            "name": "miss_audio",
            "output_subdir": "noniid_a1p0_c36_pt_fast_mi0_ma{}_mt0".format(miss_tag),
            "dirichlet_alpha": 1.0,
            "missing_image": 0.0,
            "missing_audio": float(missing_rate),
            "missing_text": 0.0,
        },
        {
            "name": "miss_text",
            "output_subdir": "noniid_a1p0_c36_pt_fast_mi0_ma0_mt{}".format(miss_tag),
            "dirichlet_alpha": 1.0,
            "missing_image": 0.0,
            "missing_audio": 0.0,
            "missing_text": float(missing_rate),
        },
    ]


def parse_str_list(raw):
    values = []
    for t in str(raw).split(","):
        t = t.strip()
        if t:
            values.append(t)
    return values


def is_partition_ready(out_dir):
    required = ["train.json", "val.json", "test.json", "meta.json"]
    return all((out_dir / name).exists() for name in required)


def build_cmd(args, prepare_script, task):
    cmd = [
        args.python_exe,
        str(prepare_script),
        "--index",
        args.index,
        "--out_dir",
        args.out_dir,
        "--mode",
        "noniid",
        "--dirichlet_alpha",
        str(float(task["dirichlet_alpha"])),
        "--clients",
        str(int(args.clients)),
        "--seed",
        str(int(args.seed)),
        "--feature_backend",
        "pretrained",
        "--device",
        args.device,
        "--clip_model_name",
        args.clip_model_name,
        "--clap_model_name",
        args.clap_model_name,
        "--local_files_only",
        str(int(args.local_files_only)),
        "--pretrained_batch_size",
        str(int(args.pretrained_batch_size)),
        "--pretrained_audio_batch_size",
        str(int(args.pretrained_audio_batch_size)),
        "--missing_apply_to",
        args.missing_apply_to,
        "--missing_image",
        str(float(task["missing_image"])),
        "--missing_audio",
        str(float(task["missing_audio"])),
        "--missing_text",
        str(float(task["missing_text"])),
        "--output_subdir",
        task["output_subdir"],
    ]
    return cmd


def resolve_repo_root(repo_root_arg):
    if repo_root_arg:
        return Path(repo_root_arg).resolve()
    return Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description="Prepare AVE pretrained partitions for alpha and modality-missing settings")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--repo_root", type=str, default="")
    parser.add_argument("--index", type=str, default="./data/ave/index.json")
    parser.add_argument("--out_dir", type=str, default="./data/ave_fed")
    parser.add_argument("--clients", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--clip_model_name", type=str, default="./local_models/clip-vit-base-patch32")
    parser.add_argument("--clap_model_name", type=str, default="./local_models/clap-htsat-unfused")
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--pretrained_batch_size", type=int, default=128)
    parser.add_argument("--pretrained_audio_batch_size", type=int, default=24)
    parser.add_argument("--missing_apply_to", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    prepare_script = repo_root / "scripts" / "prepare_ave_partitions.py"
    if not prepare_script.exists():
        raise FileNotFoundError("PREPARE_SCRIPT_NOT_FOUND: {}".format(prepare_script))

    tasks = build_tasks(args.missing_rate)
    if args.only:
        wanted = set(parse_str_list(args.only))
        unknown = sorted(wanted - {t["name"] for t in tasks})
        if unknown:
            raise ValueError("UNKNOWN_TASKS: {}".format(",".join(unknown)))
        tasks = [t for t in tasks if t["name"] in wanted]
    if not tasks:
        raise ValueError("NO_TASKS_TO_RUN")

    env = os.environ.copy()
    env["ALL_PROXY"] = ""
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""

    summary = []
    print("PREPARE_START=1")
    print("REPO_ROOT={}".format(repo_root))
    print("TASKS={}".format([t["name"] for t in tasks]))

    for idx, task in enumerate(tasks, start=1):
        out_dir = (repo_root / args.out_dir / task["output_subdir"]).resolve()
        item = {
            "task": task["name"],
            "output_subdir": task["output_subdir"],
            "output_dir": str(out_dir),
            "status": "unknown",
            "elapsed_sec": 0.0,
            "cmd": None,
            "returncode": None,
        }
        if int(args.skip_existing) == 1 and is_partition_ready(out_dir):
            item["status"] = "skipped_exists"
            summary.append(item)
            print("SKIP {}/{} TASK={} PATH={}".format(idx, len(tasks), task["name"], out_dir))
            continue

        cmd = build_cmd(args, prepare_script, task)
        item["cmd"] = " ".join(cmd)
        print("RUN {}/{} TASK={}".format(idx, len(tasks), task["name"]))
        print("CMD={}".format(item["cmd"]))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
        elapsed = time.time() - t0
        item["elapsed_sec"] = float(elapsed)
        item["returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            item["status"] = "failed"
            summary.append(item)
            raise RuntimeError("TASK_FAILED={} returncode={}".format(task["name"], proc.returncode))
        if not is_partition_ready(out_dir):
            item["status"] = "failed_missing_outputs"
            summary.append(item)
            raise RuntimeError("TASK_OUTPUT_INCOMPLETE={} out_dir={}".format(task["name"], out_dir))

        item["status"] = "ok"
        summary.append(item)
        print("DONE TASK={} ELAPSED={:.2f}s".format(task["name"], elapsed))

    print("SUMMARY_ITEMS={}".format(len(summary)))
    print("PREPARE_DONE=1")


if __name__ == "__main__":
    main()
