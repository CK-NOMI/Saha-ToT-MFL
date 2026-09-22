# Staleness-Aware Hybrid Aggregation with Tree-of-Thoughts Scheduling for Multimodal Federated Learning

## 1. Paper Information

This repository is the official code release and reproducibility entry for:

**Staleness-Aware Hybrid Aggregation with Tree-of-Thoughts Scheduling for Multimodal Federated Learning**

Authors: Yihao Pan, Kang Chen, Yang Li, Zihao Wu, Huanlai Xing, Yuan Wu.

## 2. Method Overview

This work targets multimodal hierarchical federated learning under non-IID data, modality missingness, and heterogeneous regional latency.

The proposed method combines:
- **Staleness-aware hybrid aggregation**: synchronous aggregation within gateway regions and asynchronous aggregation across gateways with staleness decay.
- **MMQS scheduling**: six-dimensional multimodal client scoring.
- **ToTs dynamic weighting**: adaptive MMQS weight allocation through ToTs reasoning.
- **Loss-aware dynamic Top-K**: dynamic participant budget control during training.

The current repository requires that the **AVE pre-trained feature slices** be prepared correctly. By default, the project uses the `AVE_pt_fast` style of data slicing. Please refer to the section on generating pre-trained feature slices.

## 3. Repository Structure

```text
.
|- configs/AVE/        # AVE experiment configurations
|- models/AVE/         # AVE model definitions
|- server/             # sync/async/hybrid servers and MMQS scheduling
|- scripts/            # data preparation, API test, and experiment scripts
|  `- data/build_ave_dataset.py  # optional raw-AVE to processed-AVE utility
|- utils/              # shared utility helpers
|- load_data.py        # multimodal data loading entry
|- delays/             # delay profiles
|- run.py              # single-run entry
`- requirements.txt    # dependencies
```


## 4. Environment Setup

Recommended environment:
- Python 3.7
- PyTorch 1.11.0 (with matching `torchvision` / `torchaudio`)
- CUDA-capable GPU

Other dependencies (e.g., `transformers`, `open-clip-torch`, `Pillow`, `scikit-learn`, `pandas`, `requests`) are installed via `requirements.txt`.

```bash
conda create -n mmqs-py37 python=3.7 -y
conda activate mmqs-py37
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## 5. AVE Data Preparation and Feature-Based Partitioning

### AVE Release for Reproducibility

To reduce cross-machine variance, we use a fixed processed AVE release in our experiments.

Due to differences in FFmpeg/JPEG/WAV decoding and encoding pipelines across operating systems and library versions, rebuilding AVE from raw videos may produce binary-different media files, even when labels and split definitions are identical.

### Download Processed AVE Release (Required)

Before running experiments, please download our processed AVE release:

https://drive.google.com/drive/folders/1DIF2Lu-cUEqrUbOBvvb9jPikch18aajp?usp=sharing

After download, place it at:

`<repo_root>/data/ave/`

Expected structure:

```text
data/ave/
|- images/
|- audio/
|- texts/
`- index.json
```

Default path used by this repository:
- AVE index path: `./data/ave/index.json`

### Optional Utility: `scripts/data/build_ave_dataset.py`

This utility converts raw AVE videos plus official split files into the repository-ready multimodal format (`images/`, `audio/`, `texts/`, `index.json`). The text modality is a label-independent natural-language caption generated from the representative frame only:

```text
AVE video
  -> ffmpeg representative frame
  -> VLM generates a natural-language caption from the frame only
  -> CLIP encodes image and text

AVE video
  -> ffmpeg audio segment
  -> CLAP encodes audio
```

Caption generation does not receive a class name, label, event category, ground-truth annotation, filename, or metadata. Captions can be supplied from an existing frame-caption file:

```bash
python scripts/data/build_ave_dataset.py \
  --captions-json ./captions_from_frames.json
```

or generated through an OpenAI-compatible VLM endpoint:

```bash
export VLM_API_KEY="..."
python scripts/data/build_ave_dataset.py \
  --vlm-api-url <endpoint> \
  --vlm-model <vision-model> \
  --vlm-api-key-env VLM_API_KEY
```

There is no label-derived caption fallback.

For reproducibility of the reported results, you can directly use the processed AVE release from the link above. In that case, you do **not** need to run this utility.

Statistics of the processed AVE release used in this work:
- Total samples: 4,143
- Train / Val / Test: 3,339 / 402 / 402
- Image files: 4,143
- Audio files: 4,143
- Text files: 4,143

All records are indexed by `index.json` and referenced with dataset-relative paths.
For strict reproduction of reported results, use this processed AVE release directly.

This repository expects the AVE dataset to be prepared locally.

In this project, **pretrained feature-based partitioning** means:
1. Use pretrained encoders (CLIP for image/text, CLAP for audio) to convert each sample into a fixed feature vector `x`.
2. Optionally, modality-missing simulation can be enabled for sensitivity/ablation studies (default: all missing rates are 0).
3. Partition samples into federated clients using Dirichlet-based non-IID splitting.

This is a **data preparation step**, not federated pretraining.
Training is performed on pre-extracted feature vectors rather than end-to-end raw image/audio/text encoders. Its purpose is to make training more reproducible and directly controllable in terms of non-IID strength and modality missingness. Since our work primarily focuses on scheduling algorithms, using pre-trained models for feature extraction is a reasonable and appropriate design choice.

Expected index path:

```text
data/ave/index.json
```

Minimal `index.json` item format:

```json
{
  "id": "sample_0001",
  "image": "images/0001.jpg",
  "audio": "audio/0001.wav",
  "text": "texts/0001.txt",
  "label": 3,
  "split": "train"
}
```

Paths in `image/audio/text` are dataset-relative paths (for example, relative to `data/ave/` when `index.json` is `data/ave/index.json`).

`--mode noniid` and `--dirichlet_alpha` control non-IID partitioning strength.

Generate the default non-IID AVE partition (36 clients, pretrained feature backend):

```bash
python scripts/prepare_ave_partitions.py \
  --index ./data/ave/index.json \
  --out_dir ./data/ave_fed \
  --mode noniid \
  --dirichlet_alpha 1.0 \
  --clients 36 \
  --seed 42 \
  --feature_backend pretrained \
  --device cuda \
  --clip_model_name ./local_models/clip-vit-base-patch32 \
  --clap_model_name ./local_models/clap-htsat-unfused \
  --local_files_only 1 \
  --pretrained_batch_size 128 \
  --pretrained_audio_batch_size 24 \
  --missing_apply_to train \
  --missing_image 0 \
  --missing_audio 0 \
  --missing_text 0 \
  --output_subdir noniid_a1p0_c36_pt_fast
```

Expected output:

```text
data/ave_fed/noniid_a1p0_c36_pt_fast/
|- train.json
|- val.json
|- test.json
`- meta.json
```

The official AVE train, validation, and test splits are preserved separately. Validation data are used for online scheduling and model-selection feedback; final test data are never used for ToT branch selection or Dynamic Top-k feedback.

For sensitivity experiments, note that:
The modality-missing experiments in this repository are conducted as controlled feature-level simulations on pre-extracted multimodal representations. Specifically, the `--missing_image`, `--missing_audio`, and `--missing_text` settings are applied in the pre-extracted feature space. These experiments are intended to evaluate the robustness of the proposed scheduling and aggregation mechanisms under representation-level modality missingness, rather than to reproduce end-to-end sensor acquisition failures, transmission corruption, or encoder-level abnormalities.
## 6. Main Running Commands

Single-trial sanity check:

```bash
python scripts/experiments/ave/main8/noniid_a1p0_c36_t6/run_ave_main8.py \
  --data_path ./data/ave_fed/noniid_a1p0_c36_pt_fast \
  --methods mmqs_fw \
  --trials 0
```

This fixed-weight path does not require ToTs API configuration.

Full method (Hybrid + MMQS + ToTs dynamic weighting + dynamic Top-K):

```bash
python scripts/experiments/ave/main8/noniid_a1p0_c36_t6/run_ave_main8.py \
  --data_path ./data/ave_fed/noniid_a1p0_c36_pt_fast \
  --methods mmqs \
  --trials 0
```

This command requires ToTs API configuration; see Section 7.

### Implementation clarification: ToT branch evaluation

For each valid ToT candidate branch, the implementation independently selects clients, performs real local training and edge FedAvg from the same gateway base model, and evaluates the resulting branch model on the validation set. The resulting validation accuracy is the empirical reward for that branch. Proxy rewards are used only for preliminary candidate assessment and do not replace empirical branch rewards.

The final test set is evaluated only after training and is not fed back into ToT pruning, client scheduling, Dynamic Top-k, early stopping, or checkpoint selection.

### Default Hyperparameters (Main AVE Runs)

For main AVE runs (excluding sensitivity sweeps), the default settings are:

- `mu = 1.5`: staleness decay exponent in polynomial staleness weighting.
- `beta = 5`: loss-aware dynamic Top-K control strength.
- `w = 10`: sliding window size (`mmqs_W`) used by loss-aware dynamic Top-K control.

These defaults are used for the main configuration unless a sensitivity script explicitly overrides them.


## 7. ToTs API Configuration (Required for Full MMQS)

This repository does not provide a default online API endpoint. If you want to enable ToT/LLM online reasoning, you must prepare your own available OpenAI-compatible API endpoint, model name, and API key.

In addition:
- This repository does not bundle any online inference service and does not include any ready-to-use API account, key, URL, or model configuration.
- This repository is not responsible for third-party API purchasing, billing, quota limits, stability, or service quality. Any related cost is borne by the user.
- Different vendors may implement OpenAI-compatible APIs with different request fields, streaming behaviors, authentication methods, and response formats. This repository only provides a generic integration entry and does not guarantee out-of-the-box compatibility with every third-party API dialect.

For ToTs-based methods (`mmqs`, `mmqs_wo_mc`), you must manually configure API settings before running.

- Provide a valid API URL, model name, and API-key environment variable.
- In `run_ave_main8.py`, selecting these methods enables ToTs weighting automatically.
- Use `run.py` with explicit API arguments:

```bash
python run.py \
  -c ./configs/AVE/hybrid_noniid_ave_tuned.json \
  -sel mmqs \
  --mmqs_enabled \
  --mmqs_weight_mode tot_api \
  --mmqs_tot_api_enabled \
  --mmqs_tot_api_url <your_api_url> \
  --mmqs_tot_api_model <your_model_name> \
  --mmqs_tot_api_key_env <your_api_key_env>
```

Here, `--mmqs_tot_api_key_env` should be the name of the environment variable that stores your API key, not the key value itself.

Example:

```bash
export MMQS_TOT_API_KEY="your_api_key"
```

Then set:

```bash
--mmqs_tot_api_key_env MMQS_TOT_API_KEY
```

Optional connectivity check:

```bash
python scripts/test_tot_api.py \
  --api_url <your_api_url> \
  --model <your_model_name> \
  --api_key_env <your_api_key_env>
```

## 8. Experiment Scripts

| Script | Purpose |
| --- | --- |
| `scripts/experiments/ave/main8/noniid_a1p0_c36_t6/run_ave_main8.py` | Main AVE experiments (core methods and primary ablations). |
| `scripts/experiments/ave/sensitivity/alpha_noniid_c36_t6/run_ave_alpha_sensitivity.py` | Dirichlet-alpha data-heterogeneity sensitivity. |
| `scripts/experiments/ave/sensitivity/mu_center1p0_t6/run_ave_mu_sensitivity.py` | Staleness `mu` sensitivity. |
| `scripts/experiments/ave/sensitivity/window_w_t6/run_ave_window_sensitivity.py` | MMQS window sensitivity. |
| `scripts/experiments/ave/sensitivity/kfit_beta_t6/run_ave_kfit_sensitivity.py` | Dynamic Top-K beta sensitivity. |
| `scripts/experiments/ave/sensitivity/modality_missing_t6/run_ave_modality_missing_sensitivity.py` | Modality-missing sensitivity (feature-level simulation). |
| `scripts/experiments/ave/sensitivity/modal_6d_t6/run_ave_6d_modal_main8.py` | 6D modal ablation (`full` vs `wo_mc`). |

## 9. External Baseline Scope

This repository currently provides the implementation of the proposed method (MMQS and related ablations) and its experiment scripts. It does not directly bundle the source code of the four comparison schedulers used in the paper: **Oort, PyramidFL, MFedMC, and Hics-FL**.

Rationale:
- These baselines already have public implementations; the goal of this repository is to provide the official implementation and reproducibility entry for our method, not to redistribute multiple third-party codebases.
- Public baseline repositories typically differ in dependency versions, data interfaces, training pipelines, and project structure. Directly vendoring them into this repository would introduce extra adaptation layers and reduce reproducibility boundary clarity.
- To avoid version drift, unclear provenance after secondary modifications, and ambiguous maintenance responsibility, we keep this release focused on our method only, while baseline comparisons in the paper are aligned under a unified evaluation protocol.

Therefore, to fully reproduce cross-method comparisons, please obtain the official public implementations of these baselines and align them with the protocol used in this paper, including:
- the same data partitioning,
- training rounds and client counts,
- delay settings,
- and the same evaluation metrics/reporting rules.
## 10. Timing Convention for ToTs-based Methods

To ensure reproducible and fair scheduling-cost comparison, this work does **not** include online external LLM/ToT API query latency when reporting MMQS execution time. Instead, timing is counted after API weight outputs are available. In the implementation (`run.py`), when `mmqs_weight_mode=tot_api`, cumulative external reasoning time in the client-selection stage is subtracted from total execution time.

This treatment has two motivations:
- We evaluate the effectiveness of scheduling with LLM-produced weights, rather than the network latency of a specific online service.
- Different reviewers and follow-up users may use different vendors, service quality levels, or deployment modes. Online LLM latency can vary substantially and can otherwise dominate cross-method time comparison.

Therefore, for reproducible scheduling-cost comparison, we follow a unified treatment for ToT-based methods: **exclude online API time for obtaining weights, and retain local scheduling/training execution time after weights are obtained**.
## 11. Troubleshooting

General runtime issues:
- **Data path error**: verify `--data_path` (or `paths.data`) points to an existing generated partition directory.
- **CLIP/CLAP local model error**: check local model directories and `--local_files_only` settings during partition generation.
- **Local pretrained model not found**: verify `--clip_model_name` and `--clap_model_name` paths.
- **ToTs API failure**: verify API URL/model/key-env and network access; run `scripts/test_tot_api.py` first.
- When online ToTs calls fail, MMQS falls back to cached/static weights so training can continue, but scheduling quality may degrade.
- **GPU OOM**: reduce batch sizes or use lighter preprocessing/training settings.

Partition generation common issues:
1. **`FileNotFoundError`**
- Check whether `data/ave/index.json` exists.
- Check whether files referenced by `image/audio/text` actually exist.
- Check whether paths are correctly resolved relative to the directory containing `index.json`.

2. **Pretrained model loading failure**
- In offline mode, ensure local model directories are complete and use `--local_files_only 1`.
- In online mode, use HuggingFace model names and set `--local_files_only 0`.

3. **Out of memory during feature preparation**
- Reduce `--pretrained_batch_size`.
- Reduce `--pretrained_audio_batch_size`.
- Or switch to `--device cpu`.

4. **Generated folder name does not match your experiment convention**
- Explicitly set `--output_subdir`.

5. **Feature dimension mismatch during training**
- Delete the incorrect partition and regenerate it.
- Ensure training `paths.data` points to the latest generated partition directory.
## 12. Citation, License, Contact

- **Citation**: A BibTeX entry will be added after the manuscript becomes publicly available.
- **License**: MIT License. See the `LICENSE` file in the repository root.
- **Contact**: open an issue in this repository or contact the corresponding authors listed in the manuscript.



