                     
                       

import argparse
import hashlib
import json
import os
import re
import wave

import numpy as np
from PIL import Image


def _safe_read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _stable_bucket(token, bins):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bins


def image_feature(path, bins):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
    hist, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    denom = float(np.sum(hist))
    if denom > 0:
        hist /= denom
    return hist


def audio_feature(path, bins):
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (pcm - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        pcm = pcm / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        pcm = pcm / 2147483648.0
    else:
        raise ValueError("unsupported sample width: {}".format(sample_width))

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    if pcm.size == 0:
        return np.zeros((bins,), dtype=np.float32)

    hist, _ = np.histogram(np.clip(pcm, -1.0, 1.0), bins=bins, range=(-1.0, 1.0))
    hist = hist.astype(np.float32)
    denom = float(np.sum(hist))
    if denom > 0:
        hist /= denom
    return hist


def text_feature(path, bins):
    text = _safe_read_text(path).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    vec = np.zeros((bins,), dtype=np.float32)
    if len(tokens) == 0:
        return vec
    for tk in tokens:
        vec[_stable_bucket(tk, bins)] += 1.0
    denom = float(np.sum(vec))
    if denom > 0:
        vec /= denom
    return vec


def ensure_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def normalize_partitions(client_bins, rng):
                                                    
    empty = [i for i, b in enumerate(client_bins) if len(b) == 0]
    if len(empty) <= 0:
        return client_bins

    for cid in empty:
        sizes = [len(b) for b in client_bins]
        donor = int(np.argmax(sizes))
        if sizes[donor] <= 1:
            break
        take_idx = int(rng.randint(0, len(client_bins[donor])))
        client_bins[cid].append(client_bins[donor].pop(take_idx))
    return client_bins


def partition_iid(items, num_clients, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    chunks = np.array_split(order, num_clients)
    return [list(map(int, c.tolist())) for c in chunks]


def partition_dirichlet(items, num_clients, alpha, rng):
    by_label = {}
    for idx, item in enumerate(items):
        lb = int(item["label"])
        by_label.setdefault(lb, []).append(idx)

    client_bins = [[] for _ in range(num_clients)]
    for _, label_indices in sorted(by_label.items(), key=lambda kv: kv[0]):
        label_indices = list(label_indices)
        rng.shuffle(label_indices)
        probs = rng.dirichlet(alpha * np.ones(num_clients))
        counts = rng.multinomial(len(label_indices), probs)
        cursor = 0
        for cid, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            client_bins[cid].extend(label_indices[cursor: cursor + cnt])
            cursor += cnt

    client_bins = normalize_partitions(client_bins, rng)
    for cid in range(num_clients):
        rng.shuffle(client_bins[cid])
    return client_bins


def make_leaf_payload(users, items, partitions):
    payload = {"users": users, "num_samples": [], "user_data": {}}
    for cid, user in enumerate(users):
        idxs = partitions[cid]
        x = [items[i]["x"] for i in idxs]
        y = [int(items[i]["label"]) for i in idxs]
        payload["num_samples"].append(len(x))
        if len(idxs) > 0:
            avail_mean = float(np.mean([items[i]["available_modalities"] for i in idxs]))
        else:
            avail_mean = 3.0
        total_modalities = 3.0
        missing_modalities = max(0.0, total_modalities - avail_mean)
        missing_ratio = missing_modalities / total_modalities
        payload["user_data"][user] = {
            "x": x,
            "y": y,
            "total_modalities": total_modalities,
            "available_modalities": avail_mean,
            "missing_modalities": missing_modalities,
            "missing_modal_ratio": missing_ratio,
            "s_modal": avail_mean / total_modalities
        }
    return payload


def maybe_drop_modalities(rec, split, rng, args):
    apply_flag = (args.missing_apply_to == "all") or (split == "train")
    drop_image = apply_flag and (rng.rand() < args.missing_image)
    drop_audio = apply_flag and (rng.rand() < args.missing_audio)
    drop_text = apply_flag and (rng.rand() < args.missing_text)
    return drop_image, drop_audio, drop_text


def format_float_tag(v):
    return ("{:.3f}".format(float(v))).rstrip("0").rstrip(".").replace(".", "p")


def build_output_dir(args):
    if args.output_subdir:
        return os.path.join(args.out_dir, args.output_subdir)
    mode_tag = "iid" if args.mode == "iid" else "noniid_a{}".format(format_float_tag(args.dirichlet_alpha))
    backend_tag = "_pt_fast" if str(args.feature_backend).strip().lower() == "pretrained" else ""
    base = "{}_c{}{}".format(mode_tag, int(args.clients), backend_tag)
    if args.missing_image > 0 or args.missing_audio > 0 or args.missing_text > 0:
        base += "_mi{}_ma{}_mt{}".format(
            format_float_tag(args.missing_image),
            format_float_tag(args.missing_audio),
            format_float_tag(args.missing_text)
        )
    return os.path.join(args.out_dir, base)


def main():
    parser = argparse.ArgumentParser(description="Prepare AVE federated IID/Non-IID partitions.")
    parser.add_argument("--index", type=str, default="./data/ave/index.json",
                        help="Path to AVE index.json")
    parser.add_argument("--out_dir", type=str, default="./data/ave_fed",
                        help="Output root directory")
    parser.add_argument("--output_subdir", type=str, default="",
                        help="Optional fixed output subdir name")
    parser.add_argument("--mode", type=str, default="noniid", choices=["iid", "noniid"],
                        help="Partition mode")
    parser.add_argument("--clients", type=int, default=36,
                        help="Number of federated clients/users")
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0,
                        help="Dirichlet alpha for non-IID partition")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--image_bins", type=int, default=32,
                        help="Feature bins for image histogram")
    parser.add_argument("--audio_bins", type=int, default=16,
                        help="Feature bins for audio histogram")
    parser.add_argument("--text_bins", type=int, default=48,
                        help="Feature bins for text hashing")
    parser.add_argument("--missing_apply_to", type=str, default="train", choices=["train", "all"],
                        help="Which split receives modality missingness simulation")
    parser.add_argument("--missing_image", type=float, default=0.0,
                        help="Missing rate for image modality")
    parser.add_argument("--missing_audio", type=float, default=0.0,
                        help="Missing rate for audio modality")
    parser.add_argument("--missing_text", type=float, default=0.0,
                        help="Missing rate for text modality")
    parser.add_argument("--feature_backend", type=str, default="pretrained",
                        choices=["handcrafted", "pretrained"],
                        help="Feature backend: handcrafted hist/hash or pretrained encoders")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for pretrained encoders")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model name for image/text pretrained features")
    parser.add_argument("--clap_model_name", type=str, default="laion/clap-htsat-unfused",
                        help="CLAP model name for audio pretrained features")
    parser.add_argument("--local_files_only", type=int, default=0,
                        help="1 means only load local cached/model-dir files for pretrained backend")
    parser.add_argument("--pretrained_batch_size", type=int, default=32,
                        help="Batch size used for CLIP image/text encoding in pretrained mode")
    parser.add_argument("--pretrained_audio_batch_size", type=int, default=8,
                        help="Batch size used for CLAP audio encoding in pretrained mode")
    args = parser.parse_args()

    if args.clients <= 0:
        raise ValueError("clients must be > 0")
    if args.mode == "noniid" and args.dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0 in noniid mode")
    for name, value in (
        ("missing_image", args.missing_image),
        ("missing_audio", args.missing_audio),
        ("missing_text", args.missing_text),
    ):
        if value < 0 or value > 1:
            raise ValueError("{} must be in [0,1]".format(name))
    if args.pretrained_batch_size <= 0:
        raise ValueError("pretrained_batch_size must be > 0")
    if args.pretrained_audio_batch_size <= 0:
        raise ValueError("pretrained_audio_batch_size must be > 0")

    rng = np.random.RandomState(args.seed)
    index_path = os.path.abspath(args.index)
    ave_root = os.path.dirname(index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    feature_builder = None
    if args.feature_backend == "pretrained":
        from pretrained_feature_utils import AVEPretrainedFeatureBuilder

        feature_builder = AVEPretrainedFeatureBuilder(
            clip_model_name=args.clip_model_name,
            clap_model_name=args.clap_model_name,
            device=args.device,
            local_files_only=bool(args.local_files_only),
            audio_fallback_bins=args.audio_bins
        )
        image_zero = feature_builder.image_zero
        audio_zero = feature_builder.audio_zero
        text_zero = feature_builder.text_zero
    else:
        image_zero = np.zeros((args.image_bins,), dtype=np.float32)
        audio_zero = np.zeros((args.audio_bins,), dtype=np.float32)
        text_zero = np.zeros((args.text_bins,), dtype=np.float32)

    samples_train = []
    samples_val = []
    samples_test = []

    prepared = []
    for ridx, rec in enumerate(records):
        split = str(rec.get("split", "")).strip().lower()
        label = int(rec["label"])
        sample_id = str(rec.get("id", "sample_{}".format(ridx)))

        image_path = os.path.join(ave_root, rec["image"])
        audio_path = os.path.join(ave_root, rec["audio"])
        text_path = os.path.join(ave_root, rec["text"])
        ensure_exists(image_path)
        ensure_exists(audio_path)
        ensure_exists(text_path)

        drop_image, drop_audio, drop_text = maybe_drop_modalities(rec, split, rng, args)
        prepared.append({
            "split": split,
            "label": label,
            "id": sample_id,
            "image_path": image_path,
            "audio_path": audio_path,
            "text_path": text_path,
            "drop_image": bool(drop_image),
            "drop_audio": bool(drop_audio),
            "drop_text": bool(drop_text)
        })

    if args.feature_backend == "pretrained":
        total = len(prepared)
        step = int(max(1, args.pretrained_batch_size))
        for start in range(0, total, step):
            end = min(total, start + step)
            chunk = prepared[start:end]
            records_batch = [(x["image_path"], x["audio_path"], x["text_path"]) for x in chunk]
            image_mat, audio_mat, text_mat = feature_builder.encode_batch(
                records_batch,
                clip_batch_size=args.pretrained_batch_size,
                clap_batch_size=args.pretrained_audio_batch_size
            )

            for i, meta in enumerate(chunk):
                image_vec = image_mat[i]
                audio_vec = audio_mat[i]
                text_vec = text_mat[i]
                if meta["drop_image"]:
                    image_vec = image_zero
                if meta["drop_audio"]:
                    audio_vec = audio_zero
                if meta["drop_text"]:
                    text_vec = text_zero

                available = 3 - int(meta["drop_image"]) - int(meta["drop_audio"]) - int(meta["drop_text"])
                feat = np.concatenate([image_vec, audio_vec, text_vec], axis=0).astype(np.float32)
                item = {
                    "id": meta["id"],
                    "split": meta["split"],
                    "label": meta["label"],
                    "x": feat.tolist(),
                    "available_modalities": float(available)
                }
                if meta["split"] == "train":
                    samples_train.append(item)
                elif meta["split"] == "val":
                    samples_val.append(item)
                elif meta["split"] == "test":
                    samples_test.append(item)

            if ((start // step) % 20 == 0) or (end == total):
                print("PT_PROGRESS_AVE={}/{}".format(end, total))
    else:
        for meta in prepared:
            image_vec = image_feature(meta["image_path"], args.image_bins)
            audio_vec = audio_feature(meta["audio_path"], args.audio_bins)
            text_vec = text_feature(meta["text_path"], args.text_bins)

            if meta["drop_image"]:
                image_vec = image_zero
            if meta["drop_audio"]:
                audio_vec = audio_zero
            if meta["drop_text"]:
                text_vec = text_zero

            available = 3 - int(meta["drop_image"]) - int(meta["drop_audio"]) - int(meta["drop_text"])
            feat = np.concatenate([image_vec, audio_vec, text_vec], axis=0).astype(np.float32)
            item = {
                "id": meta["id"],
                "split": meta["split"],
                "label": meta["label"],
                "x": feat.tolist(),
                "available_modalities": float(available)
            }
            if meta["split"] == "train":
                samples_train.append(item)
            elif meta["split"] == "val":
                samples_val.append(item)
            elif meta["split"] == "test":
                samples_test.append(item)

    users = ["user_{:04d}".format(i) for i in range(args.clients)]
    if args.mode == "iid":
        train_partitions = partition_iid(samples_train, args.clients, rng)
        val_partitions = partition_iid(samples_val, args.clients, rng)
        test_partitions = partition_iid(samples_test, args.clients, rng)
    else:
        train_partitions = partition_dirichlet(samples_train, args.clients, args.dirichlet_alpha, rng)
        val_partitions = partition_dirichlet(samples_val, args.clients, args.dirichlet_alpha, rng)
        test_partitions = partition_dirichlet(samples_test, args.clients, args.dirichlet_alpha, rng)

    train_payload = make_leaf_payload(users, samples_train, train_partitions)
    val_payload = make_leaf_payload(users, samples_val, val_partitions)
    test_payload = make_leaf_payload(users, samples_test, test_partitions)

    out_path = build_output_dir(args)
    os.makedirs(out_path, exist_ok=True)
    train_file = os.path.join(out_path, "train.json")
    val_file = os.path.join(out_path, "val.json")
    test_file = os.path.join(out_path, "test.json")
    meta_file = os.path.join(out_path, "meta.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_payload, f, ensure_ascii=False)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_payload, f, ensure_ascii=False)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_payload, f, ensure_ascii=False)

    train_counts = train_payload["num_samples"]
    val_counts = val_payload["num_samples"]
    test_counts = test_payload["num_samples"]
    feature_dim = 0
    if len(samples_train) > 0:
        feature_dim = int(len(samples_train[0]["x"]))
    elif len(samples_val) > 0:
        feature_dim = int(len(samples_val[0]["x"]))
    elif len(samples_test) > 0:
        feature_dim = int(len(samples_test[0]["x"]))
    elif args.feature_backend == "pretrained":
        feature_dim = int(feature_builder.feature_dim)
    else:
        feature_dim = int(args.image_bins + args.audio_bins + args.text_bins)
    meta = {
        "index": index_path,
        "output_dir": out_path,
        "mode": args.mode,
        "clients": args.clients,
        "dirichlet_alpha": args.dirichlet_alpha if args.mode == "noniid" else None,
        "seed": args.seed,
        "feature_dim": feature_dim,
        "num_classes": len(sorted(list(set([int(x["label"]) for x in samples_train])))),
        "train_samples": len(samples_train),
        "val_samples": len(samples_val),
        "test_samples": len(samples_test),
        "train_min_per_client": int(min(train_counts)) if len(train_counts) > 0 else 0,
        "train_max_per_client": int(max(train_counts)) if len(train_counts) > 0 else 0,
        "val_min_per_client": int(min(val_counts)) if len(val_counts) > 0 else 0,
        "val_max_per_client": int(max(val_counts)) if len(val_counts) > 0 else 0,
        "test_min_per_client": int(min(test_counts)) if len(test_counts) > 0 else 0,
        "test_max_per_client": int(max(test_counts)) if len(test_counts) > 0 else 0,
        "missing_apply_to": args.missing_apply_to,
        "missing_image": args.missing_image,
        "missing_audio": args.missing_audio,
        "missing_text": args.missing_text,
        "feature_backend": args.feature_backend,
        "clip_model_name": args.clip_model_name if args.feature_backend == "pretrained" else None,
        "clap_model_name": args.clap_model_name if args.feature_backend == "pretrained" else None,
        "local_files_only": int(args.local_files_only),
        "pretrained_batch_size": int(args.pretrained_batch_size),
        "pretrained_audio_batch_size": int(args.pretrained_audio_batch_size)
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("AVE_PREP_OK=1")
    print("OUT_DIR={}".format(out_path))
    print("TRAIN_JSON={}".format(train_file))
    print("VAL_JSON={}".format(val_file))
    print("TEST_JSON={}".format(test_file))
    print("META_JSON={}".format(meta_file))
    print(
        "SUMMARY mode={} clients={} train={} val={} test={} feature_dim={} classes={}".format(
            args.mode, args.clients, len(samples_train), len(samples_val), len(samples_test), feature_dim, meta["num_classes"]
        )
    )


if __name__ == "__main__":
    main()
