# MeDAC Reproduction Guide

This document explains how to reproduce the MeDAC experiments end-to-end, including environment setup, data generation, data preparation, model training/evaluation, and speedup reporting.

## 1. Scope

This repository contains four scenario folders:

- `nnsmith_ort_v1.16.0/nnsmith`
- `nnsmith_ort_v1.17.0/nnsmith`
- `nnsmith_tvm_v0.10.0/nnsmith`
- `nnsmith_tvm_v0.11.1/nnsmith`

Each scenario includes:

- NNSmith-based model generation and execution scripts
- Data processing scripts (`error.py`, `error_classify.py`, `error_classify_op.py`, `merge_info*.py`)
- Learning-based prioritization notebooks (`MPNN_edge_message_TCP.ipynb`, `LET.ipynb`, `gcn_new.ipynb`)
- Saved model checkpoints (`*.pth`) and speedup artifacts (`*.pkl`)

## 2. Recommended Environment

The original experiments were run in the following environment:

- GPU: NVIDIA RTX 4080 (16GB)
- CUDA: 12.3
- OS: Ubuntu 22.04 LTS
- Framework: PyTorch

Even though your current machine may be Windows, the scripts in this project (`*.sh`) are Linux shell scripts. Reproduction is recommended on Ubuntu/Linux.

## 3. Environment Setup

From repository root:

```bash
cd MeDAC-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `requirements.txt` already includes key packages such as `torch`, `onnx`, `onnxruntime`, `apache-tvm`, and `nnsmith`.
- If you use a different CUDA/PyTorch build, align package versions manually.

## 4. Reproduction Modes

You can reproduce results in two modes.

### Mode A: Fast Reproduction (recommended first)

Use existing prepared artifacts and checkpoints in each scenario folder.

1. Open scenario notebook(s):
   - `MPNN_edge_message_TCP.ipynb`
   - `LET.ipynb`
   - `gcn_new.ipynb`
2. Run cells for:
   - loading prepared test datasets (`*-testdata_*` folders)
   - loading model checkpoint (`*.pth`)
   - computing prioritized execution order
   - computing speedup curves and summary stats
3. Export/verify generated figures and speedup outputs (`*.pkl`).

### Mode B: Full Rebuild (data generation to speedup)

Run all steps below for one or more scenario folders.

## 5. Full Rebuild Workflow (Per Scenario)

Assume scenario root is:

```bash
cd nnsmith_ort_v1.17.0/nnsmith
```

Replace with other scenario paths as needed.

### Step 1. Generate random ONNX models

Run:

```bash
bash generate_model.sh
```

Script behavior:

- loops over a large range (`0..249999` or `0..299999`, scenario-dependent)
- samples `mgen.max_nodes` in `[1, 30]`
- runs `nnsmith.model_gen model.type=onnx debug.viz=true ...`
- stores each model in `model_i` folders

### Step 2. Execute generated models and record execution metadata

Run:

```bash
bash exec_model.sh
```

Script behavior:

- iterates all `model_i` folders
- runs `nnsmith.model_exec` with scenario backend
- writes execution report/bug report under model folder

### Step 3. Separate error models

Run scripts in this order:

```bash
python error.py
python error_classify.py
python error_classify_op.py
```

What they do:

- `error.py`: move models with `bug_report/` into error bucket
- `error_classify.py`: separate non-accuracy errors from accuracy mismatch
- `error_classify_op.py`: optionally separate unsupported-operator related errors

### Step 4. Build training dataset

Run:

```bash
python merge_info_train.py
```

Typical behavior:

- sample balanced train set from normal and error pools
- output training JSON (for example `train_3000_3000.json` or `train_1000_1000.json`)

### Step 5. Build evaluation/test dataset for speedup simulation

Run:

```bash
python merge_info.py
```

Typical behavior:

- sample many normal models + very few bug models
- output scenario test files such as:
  - `*-testdata_0_7/demo_test_data_info_0.json`
  - `*-testdata_0_9/demo_test_data_info_0.json`
  - `*-testdata_0_11/demo_test_data_info_0.json`

### Step 6. Train and evaluate prioritization models

Open and run notebooks:

- `MPNN_edge_message_TCP.ipynb` (MeDAC core model)
- `LET.ipynb` (LET baseline)
- `gcn_new.ipynb` (GCN baseline)

Outputs include:

- trained checkpoints (`best_model_0.pth`, `ablation_best_model.pth`, etc.)
- bug-detection time traces (`mpnn-*.pkl`, `let-*.pkl`, `gcn-*.pkl`)
- speedup summaries (`*-speedup.pkl`)

### Step 7. Aggregate cross-version and cross-compiler results

In `MPNN_edge_message_TCP.ipynb` (for representative scenario), run cells that:

- load four scenario speedup files
- compute cross-version and cross-compiler aggregates
- compute mean/median speedup
- plot distribution charts (bar/violin/hist)

## 6. Scenario Mapping

Generation and execution backends by folder:

- `nnsmith_ort_v1.16.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_ort_v1.17.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_tvm_v0.10.0/nnsmith`: `backend.type=tvm`
- `nnsmith_tvm_v0.11.1/nnsmith`: `backend.type=tvm`

## 7. Practical Notes on Folder Names

Some scripts use different folder names for generation vs execution (for example generated folder might be `onnx_models_gen_ort-v1.17.0_1`, while execution script may read `normal_models_1`).

If needed, normalize names before next step:

```bash
mv onnx_models_gen_ort-v1.17.0_1 normal_models_1
```

Use analogous renaming in other scenarios.

## 8. Core Metrics

The project focuses on bug-finding efficiency rather than only classification accuracy.

Main pipeline metric:

1. Sort test cases by model-predicted bug probability.
2. Simulate execution in this order and accumulate execution time (`texec`).
3. Record cumulative time to detect each bug.
4. Compare against random order baseline (RO).
5. Compute speedup ratio used in notebooks.

## 9. Minimal Verification Checklist

For each scenario, verify:

- training JSON exists (`train_*.json`)
- test JSON exists (`*-testdata_*/demo_test_data_info_0.json`)
- model checkpoint exists (`*.pth`)
- speedup artifact exists (`*-speedup.pkl`)
- notebook charts can be regenerated

## 10. Typical End-to-End Order (Recommended)

For one scenario:

1. `generate_model.sh`
2. `exec_model.sh`
3. `error.py`
4. `error_classify.py`
5. `error_classify_op.py`
6. `merge_info_train.py`
7. `merge_info.py`
8. Run `LET.ipynb`
9. Run `gcn_new.ipynb`
10. Run `MPNN_edge_message_TCP.ipynb`

For full paper-style comparison:

1. Repeat above for all four scenarios.
2. Run aggregate speedup cells in `MPNN_edge_message_TCP.ipynb`.
3. Export final cross-version/cross-compiler figures.

## 11. Troubleshooting

- If `nnsmith.model_gen` command is missing:
  - ensure `nnsmith` is installed in current environment.
- If backend import errors occur (`onnxruntime`/`tvm`):
  - verify package versions from `requirements.txt`.
- If notebook cells cannot find data files:
  - check scenario folder and dataset folder names.
- If shell scripts fail on Windows:
  - run in WSL or Linux environment.
