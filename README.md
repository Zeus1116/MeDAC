## MeDAC

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-a5fed.svg"></a> 

This repository provides the tool (i.e., MeDAC) and all experimental data for our work: "Accelerating Deep Learning Compiler Testing via
Message Passing Neural Network with Attention", which has been submitted to IEEE Transactions on Reliability. 

### File Structure
The code repository's root directory contains four subfolders, each corresponding to one of the four test scenarios mentioned in the paper.
Each project directory includes the following main items:
* **NNSmith**: The NNSmith folder contains the original NNSmith code, along with our implementation for logging three types of features of the generated deep learning models, as well as the generation time of each model.
* **MPNN_edge_message_TCP.ipynb**: The file MPNN_edge_message_TCP.ipynb contains the core implementation of the MeDAC approach. This Jupyter notebook provides a detailed demonstration of:(1) How we construct and train a message-passing neural network (MPNN) with an attention mechanism, along with the iterative process of updating the intermediate representations of the deep learning model by refining node and edge features. (2) How to load our preprocessed deep learning model data to accelerate the testing process of DL compilers, along with a performance evaluation of the achieved speedup.
* **LET.ipynb**: The LET.ipynb file demonstrates how we construct a deep learning model based on the LET concept to accelerate the testing process of deep learning compilers, along with the resulting speedup performance.
* **GCN.ipynb**: Similarly, the GCN.ipynb file demonstrates how we construct a deep learning model based on the GCN to accelerate the testing process of deep learning compilers, along with the resulting speedup performance.
* **Other files**: Additionally, each subfolder contains numerous auxiliary files, which include both the training datasets and the implementation details of how we utilized the NNSmith tool to generate each individual model, record model generation times, and extract internal structural features of the models.

The complete experimental data and implementation code for each testing scenario enable straightforward reproduction of all speedup effects reported in our paper.

## Quick Navigation

- Parameter reference: `PARAMETER_REFERENCE.md`
- Hardware and base environment: `Hardware.md`
- Python dependencies: `requirements.txt`

## Repository Layout

This repository contains four scenario folders (two ORT versions and two TVM versions):

- `nnsmith_ort_v1.16.0/nnsmith`
- `nnsmith_ort_v1.17.0/nnsmith`
- `nnsmith_tvm_v0.10.0/nnsmith`
- `nnsmith_tvm_v0.11.1/nnsmith`

Each scenario folder includes:

- data generation scripts (`generate_model.sh`)
- execution scripts (`exec_model.sh`)
- data processing scripts (`error.py`, `error_classify.py`, `merge_info*.py`)
- model notebooks (`MPNN_edge_message_TCP.ipynb`, `LET.ipynb`, `gcn_new.ipynb`)
- checkpoints and precomputed artifacts (`*.pth`, `*.pkl`)

## Reproduction Modes

- Fast reproduction: directly run notebook evaluation with existing artifacts/checkpoints.
- Full rebuild: regenerate models, execute tests, process datasets, train/evaluate models, and compute speedup.

## Methods and Evaluation

- Main method: MeDAC (MPNN with attention-based message passing)
- Baselines: LET and GCN
- Core metric: bug-finding efficiency (time-to-detect-bugs and speedup over random order)

Detailed hyperparameters and sampling settings are documented in `PARAMETER_REFERENCE.md`.

## Full Reproduction Guide

This section provides end-to-end reproduction steps directly in `README.md`.

### 1. Scope

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

### 2. Recommended Environment

Original experiment environment:

- GPU: NVIDIA RTX 4080 (16GB)
- CUDA: 12.3
- OS: Ubuntu 22.04 LTS
- Framework: PyTorch

Note: this project contains Linux shell scripts (`*.sh`). Reproduction is recommended on Ubuntu/Linux (or WSL).

### 3. Environment Setup

From repository root:

```bash
cd MeDAC-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Reproduction Modes

#### Mode A: Fast Reproduction (recommended first)

Use prepared artifacts/checkpoints already in scenario folders:

1. Open notebooks:
	 - `MPNN_edge_message_TCP.ipynb`
	 - `LET.ipynb`
	 - `gcn_new.ipynb`
2. Run cells for:
	 - loading prepared test datasets (`*-testdata_*`)
	 - loading checkpoints (`*.pth`)
	 - ranking test cases and simulating bug detection time
	 - computing speedup statistics and drawing figures

#### Mode B: Full Rebuild (data generation to speedup)

### 5. Full Rebuild Workflow (Per Scenario)

Example scenario path:

```bash
cd nnsmith_ort_v1.17.0/nnsmith
```

#### Step 1: Generate random ONNX models

```bash
bash generate_model.sh
```

Script behavior:

- loops over large index range (`0..249999` or `0..299999`, scenario-dependent)
- samples `mgen.max_nodes` in `[1, 30]`
- runs `nnsmith.model_gen model.type=onnx debug.viz=true ...`

#### Step 2: Execute generated models

```bash
bash exec_model.sh
```

Script behavior:

- iterates all `model_i` folders
- runs `nnsmith.model_exec` with scenario backend
- writes execution and bug-report metadata per model

#### Step 3: Separate and classify error cases

```bash
python error.py
python error_classify.py
python error_classify_op.py
```

#### Step 4: Build training dataset

```bash
python merge_info_train.py
```

Typical output: `train_*.json` and/or `train_time_*.json`

#### Step 5: Build evaluation/test dataset

```bash
python merge_info.py
```

Typical output: `*-testdata_*/demo_test_data_info_0.json` and `demo_test_time_info_0.json`

#### Step 6: Train/evaluate prioritization models

Run notebooks:

- `MPNN_edge_message_TCP.ipynb` (MeDAC core)
- `LET.ipynb` (baseline)
- `gcn_new.ipynb` (baseline)

Typical output artifacts:

- checkpoints (`*.pth`)
- detection-time traces (`mpnn-*.pkl`, `let-*.pkl`, `gcn-*.pkl`)
- speedup summaries (`*-speedup.pkl`)

#### Step 7: Aggregate cross-version/cross-compiler results

In representative notebook cells:

- load speedup files from the four scenarios
- compute cross-version and cross-compiler aggregates
- compute mean/median speedup
- draw distribution charts (bar/violin/hist)

### 6. Scenario Mapping

Generation and execution backends:

- `nnsmith_ort_v1.16.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_ort_v1.17.0/nnsmith`: `backend.type=onnxruntime`
- `nnsmith_tvm_v0.10.0/nnsmith`: `backend.type=tvm`
- `nnsmith_tvm_v0.11.1/nnsmith`: `backend.type=tvm`

### 7. Practical Folder-Name Notes

Some scripts use different folder names between generation and execution (for example, generated folder may be `onnx_models_gen_ort-v1.17.0_1`, while `exec_model.sh` reads `normal_models_1`).

If needed, normalize names manually:

```bash
mv onnx_models_gen_ort-v1.17.0_1 normal_models_1
```

Use analogous renaming in other scenarios.

### 8. Core Metric and Speedup Logic

The key metric is bug-finding efficiency, not only static classification accuracy.

Pipeline logic:

1. Rank test cases by predicted bug probability.
2. Simulate execution in ranked order and accumulate `texec`.
3. Record cumulative time for each detected bug.
4. Compare with random-order baseline (RO).
5. Compute speedup values and summarize by scenario.

### 9. Minimal Verification Checklist

Per scenario, verify:

- training JSON exists (`train_*.json`)
- test JSON exists (`*-testdata_*/demo_test_data_info_0.json`)
- checkpoints exist (`*.pth`)
- speedup artifacts exist (`*-speedup.pkl`)
- figures can be regenerated from notebooks

### 10. Typical End-to-End Order

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

For full paper-style comparison:

1. Repeat above for all four scenarios.
2. Run aggregate speedup cells.
3. Export final cross-version/cross-compiler figures.

### 11. Troubleshooting

- If `nnsmith.model_gen` is missing:
	- ensure `nnsmith` is installed in the active environment.
- If `onnxruntime` or `tvm` import fails:
	- verify package versions in `requirements.txt`.
- If notebook cannot find data files:
	- check scenario path and dataset folder names.
- If shell scripts fail on Windows:
	- run in WSL or Linux environment.

