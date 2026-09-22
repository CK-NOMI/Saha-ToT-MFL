"""
Convert the original AVE dataset (videos + split files) into the simplified
multimodal format used by our experiments:

    data/ave/
        images/<sample>.jpg
        audio/<sample>.wav
        texts/<sample>.txt
        index.json

Each entry in index.json contains:
    {
        "id": "...",
        "split": "train|val|test",
        "image": "images/....jpg",
        "audio": "audio/....wav",
        "text": "texts/....txt",
        "label": <int>
    }

Requirements:
    - FFmpeg must be installed and available in PATH.
    - Text captions must be produced from the extracted representative frame
      only.  Supply an existing caption JSON file, or configure an
      OpenAI-compatible vision-language-model (VLM) endpoint with the command
      line arguments below.  Class names, labels, and event annotations are
      never provided to the captioning step.
    - The original dataset should be extracted to:
        data/AVE_raw/AVE_Dataset/
          ├─ AVE/            (contains *.mp4 files)
          ├─ trainSet.txt
          ├─ valSet.txt
          └─ testSet.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

OUTPUT_ROOT = Path("data/ave")
RAW_ROOT = Path("data/AVE_raw/AVE_Dataset")
VIDEO_DIR = RAW_ROOT / "AVE"

SPLIT_FILES = {
    "train": RAW_ROOT / "trainSet.txt",
    "val": RAW_ROOT / "valSet.txt",
    "test": RAW_ROOT / "testSet.txt",
}

# This is deliberately restricted to observable image content.  Do not add
# dataset class names, event labels, split information, or video metadata here.
CAPTION_PROMPT = (
    "Describe only the visible scene and actions in this image in one concise "
    "natural-language sentence. Do not infer or mention dataset labels, class "
    "names, event categories, filenames, or any metadata."
)


@dataclass
class SampleEntry:
    class_name: str
    video_id: str
    split: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.5, self.end - self.start)

    def mid_time(self) -> float:
        return self.start + self.duration / 2.0


def parse_split_file(path: Path, split: str) -> List[SampleEntry]:
    entries: List[SampleEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("&")
        if len(parts) < 5:
            continue
        class_name, video_id, _, start, end = parts[:5]
        try:
            start_f = float(start)
            end_f = float(end)
        except ValueError:
            continue
        if end_f <= start_f:
            end_f = start_f + 2.0  # ensure positive duration

        entries.append(
            SampleEntry(
                class_name=class_name.strip(),
                video_id=video_id.strip(),
                split=split,
                start=start_f,
                end=end_f,
            )
        )
    return entries


def ensure_dirs(root: Path) -> Dict[str, Path]:
    subdirs = {}
    for name in ["images", "audio", "texts"]:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def run_ffmpeg(args: List[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    subprocess.run(cmd, check=True)


def build_index(entries: List[Dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def load_caption_map(path: Path) -> Dict[str, str]:
    """Load VLM captions keyed by sample id or source video id.

    Supported JSON forms are ``{"sample_id": "caption"}`` and a list of
    objects containing ``caption`` plus either ``id``/``sample_id`` or
    ``video_id``.  The captions are expected to have been generated from the
    representative frames without ground-truth metadata.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    captions: Dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                captions[str(key)] = value.strip()
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            caption = item.get("caption")
            key = item.get("id", item.get("sample_id", item.get("video_id")))
            if key is not None and isinstance(caption, str) and caption.strip():
                captions[str(key)] = caption.strip()
    else:
        raise ValueError("captions JSON must be an object or a list of caption records")
    if not captions:
        raise ValueError("captions JSON contains no usable captions")
    return captions


def caption_from_vlm(image_path: Path, args: argparse.Namespace) -> str:
    """Generate one label-independent caption from an extracted frame."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "VLM caption generation requires the requests package; install requirements.txt first"
        ) from exc
    api_key = os.environ.get(args.vlm_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            "VLM API key is unavailable; set {} or use --captions-json".format(
                args.vlm_api_key_env
            )
        )
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": args.vlm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": args.vlm_max_tokens,
    }
    response = requests.post(
        args.vlm_api_url,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=args.vlm_timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        caption = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        raise RuntimeError("VLM response does not contain a text caption") from exc
    if not caption:
        raise RuntimeError("VLM returned an empty caption")
    return caption


def main(output_root: Path = OUTPUT_ROOT, args: argparse.Namespace = None) -> None:
    if args is None:
        raise ValueError("caption configuration is required")
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    caption_map = load_caption_map(Path(args.captions_json)) if args.captions_json else None
    if caption_map is None and not (args.vlm_api_url and args.vlm_model and args.vlm_api_key_env):
        raise ValueError(
            "Provide --captions-json, or configure --vlm-api-url, --vlm-model, and --vlm-api-key-env. "
            "There is intentionally no label-derived text fallback."
        )

    for split, file in SPLIT_FILES.items():
        if not file.exists():
            raise FileNotFoundError(f"Split file missing: {file}")

    subdirs = ensure_dirs(output_root)
    records: List[Dict] = []

    print("[INFO] Parsing split files...")
    split_entries: Dict[str, List[SampleEntry]] = {}
    class_to_idx: Dict[str, int] = {}

    for split, file in SPLIT_FILES.items():
        entries = parse_split_file(file, split)
        split_entries[split] = entries
        for entry in entries:
            if entry.class_name not in class_to_idx:
                class_to_idx[entry.class_name] = len(class_to_idx)

    print(f"[INFO] Found {len(class_to_idx)} unique classes.")

    total_entries = sum(len(v) for v in split_entries.values())
    print(f"[INFO] Preparing {total_entries} samples...")

    for split, entries in split_entries.items():
        for idx, entry in enumerate(entries, 1):
            video_path = VIDEO_DIR / f"{entry.video_id}.mp4"
            if not video_path.exists():
                print(f"[WARN] Missing video {video_path}, skipping.")
                continue

            sample_id = f"{split}_{idx:05d}"
            image_path = subdirs["images"] / f"{sample_id}.jpg"
            audio_path = subdirs["audio"] / f"{sample_id}.wav"
            text_path = subdirs["texts"] / f"{sample_id}.txt"

            try:
                # Extract representative frame
                run_ffmpeg([
                    "-ss", f"{entry.mid_time():.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    str(image_path),
                ])

                # Extract audio segment
                run_ffmpeg([
                    "-ss", f"{entry.start:.3f}",
                    "-t", f"{entry.duration:.3f}",
                    "-i", str(video_path),
                    "-ac", "1",
                    "-ar", "16000",
                    str(audio_path),
                ])
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] ffmpeg failed for {sample_id}: {exc}")
                for path in [image_path, audio_path]:
                    if path.exists():
                        path.unlink()
                continue

            if caption_map is not None:
                text_content = caption_map.get(sample_id) or caption_map.get(entry.video_id)
                if not text_content:
                    raise KeyError(
                        "No caption for {} (or source video {}) in {}".format(
                            sample_id, entry.video_id, args.captions_json
                        )
                    )
            else:
                text_content = caption_from_vlm(image_path, args)
            text_path.write_text(text_content, encoding="utf-8")

            records.append({
                "id": sample_id,
                "split": split,
                "image": f"images/{image_path.name}",
                "audio": f"audio/{audio_path.name}",
                "text": f"texts/{text_path.name}",
                "label": class_to_idx[entry.class_name],
                "class_name": entry.class_name,
            })

        print(f"[INFO] Processed {len(entries)} entries for split '{split}'.")

    build_index(records, output_root / "index.json")
    print(f"[INFO] Completed. Total usable samples: {len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AVE multimodal dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to store processed samples (default: data/ave)",
    )
    parser.add_argument(
        "--captions-json",
        type=str,
        default="",
        help="Existing label-independent VLM captions, keyed by sample id or video id.",
    )
    parser.add_argument("--vlm-api-url", type=str, default="", help="OpenAI-compatible VLM chat endpoint.")
    parser.add_argument("--vlm-model", type=str, default="", help="Vision-language model identifier.")
    parser.add_argument(
        "--vlm-api-key-env",
        type=str,
        default="",
        help="Environment-variable name containing the VLM API key.",
    )
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=96)
    args = parser.parse_args()
    main(args.output_dir, args)
