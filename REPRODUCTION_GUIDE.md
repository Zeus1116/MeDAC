# MeDAC Full Reproduction Guide

This file provides the complete end-to-end reproduction workflow for all scenarios, including model generation, execution, data construction, training, and speedup analysis.

## 1. Scope

The repository contains four scenario folders:

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

Original experiment environment:

- GPU: NVIDIA RTX 4080 (16GB)
- CUDA: 12.3
- OS: Ubuntu 22.04 LTS
- Framework: PyTorch

Note: this project contains Linux shell scripts (`*.sh`). Reproduction is recommended on Ubuntu/Linux (or WSL).

## 3. Environment Setup

From repository root:

```bash
cd MeDAC-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Reproduction Modes

### Mode A: Fast Reproduction

Use existing prepared artifacts and checkpoints:

1. Open `MPNN_edge_message_TCP.ipynb`, `LET.ipynb`, `gcn_new.ipynb`.
2. Run cells to load prepared datasets/checkpoints.
3. Run ranking and speedup evaluation cells.

### Mode B: Full Rebuild

Run the full pipeline in Section 5.

## 5. Full Rebuild Workflow (Per Scenario)

Example scenario path:

```bash
cd nnsmith_ort_v1.17.0/nnsmith
```

### Step 1: Generate random ONNX models

```bash
bash generate_model.sh
```

### Step 2: Execute generated models

```bash
bash exec_model.sh
```

### Step 3: Separate and classify error cases

```bash
python error.py
python error_classify.py
python error_classify_op.py
```

### Step 4: Build training dataset

```bash
python merge_info_train.py
```

### Step 5: Build evaluation/test dataset

```bash
python merge_info.py
```

### Step 6: Train/evaluate prioritization models

Run notebooks:

- `MPNN_edge_message_TCP.ipynb`
- `LET.ipynb`
- `gcn_new.ipynb`

### Step 7: Aggregate cross-version/cross-compiler results

In representative notebook cells:

- load speedup files from four scenarios
- compute cross-version and cross-compiler aggregates
- compute mean/median speedup
- draw distribution charts

## 6. Scenario Mapping

Generation and execution backends:

- `nnsmith_ort_v1.16.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_ort_v1.17.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_tvm_v0.10.0/nnsmith`: `backend.type=tvm`
- `nnsmith_tvm_v0.11.1/nnsmith`: `backend.type=tvm`

## 7. Practical Folder-Name Notes

Some scripts use different folder names between generation and execution (for example, generated folder may be `onnx_models_gen_ort-v1.17.0_1`, while `exec_model.sh` reads `normal_models_1`).

If needed, normalize names manually:

```bash
mv onnx_models_gen_ort-v1.17.0_1 normal_models_1
```

## 8. Core Metric and Speedup Logic

1. Rank test cases by predicted bug probability.
2. Simulate execution in ranked order and accumulate `texec`.
3. Record cumulative time for each detected bug.
4. Compare with random-order baseline (`RO`).
5. Compute speedup values and summarize by scenario.

## 9. Minimal Verification Checklist

Per scenario, verify:

- training JSON exists (`train_*.json`)
- test JSON exists (`*-testdata_*/demo_test_data_info_0.json`)
- checkpoints exist (`*.pth`)
- speedup artifacts exist (`*-speedup.pkl`)
- figures can be regenerated from notebooks

## 10. Typical End-to-End Order

For one scenario:

1. `generate_model.sh`
2. `exec_model.sh`
3. `error.py`
4. `error_classify.py`
5. `error_classify_op.py`
6. `merge_info_train.py`
7. `merge_info.py`
8. run `LET.ipynb`
9. run `gcn_new.ipynb`
10. run `MPNN_edge_message_TCP.ipynb`

For full comparison:

1. Repeat for all four scenarios.
2. Run aggregate speedup cells.
3. Export final figures.

## 11. Troubleshooting

- If `nnsmith.model_gen` is missing: ensure `nnsmith` is installed in active environment.
- If `onnxruntime` or `tvm` import fails: verify versions in `requirements.txt`.
- If notebook cannot find data files: check scenario path and dataset folder names.
- If shell scripts fail on Windows: run in WSL or Linux.
