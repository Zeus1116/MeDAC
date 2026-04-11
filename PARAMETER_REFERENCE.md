# MeDAC Parameter Reference

This file summarizes key experimental parameters used across scripts and notebooks.

## 1. Data Generation Parameters

Source scripts:

- `nnsmith_ort_v1.16.0/nnsmith/generate_model.sh`
- `nnsmith_ort_v1.17.0/nnsmith/generate_model.sh`
- `nnsmith_tvm_v0.10.0/nnsmith/generate_model.sh`
- `nnsmith_tvm_v0.11.1/nnsmith/generate_model.sh`

Common parameters:

- `model.type=onnx`
- `debug.viz=true`
- `mgen.max_nodes=$MAX_NODES`, where `MAX_NODES` is sampled in `[1, 30]`

Backend per scenario:

- ORT scenarios: `backend.type=onnxruntime`
- TVM scenarios: `backend.type=tvm`

Loop count in scripts:

- ORT scripts: `0..249999`
- TVM scripts: `0..299999`

## 2. Model Execution Parameters

Source scripts:

- `*/nnsmith/exec_model.sh`

Common execution command pattern:

```bash
nnsmith.model_exec model.type=onnx backend.type=<onnxruntime|tvm> model.path=nnsmith_output/model.onnx
```

Recorded time fields in downstream JSONs:

- generation side: `tgen`, `tmat`, `tfetch`, `tsave`
- execution side: `texec` (from `exec_time.json`)

## 3. Error Processing Rules

Source scripts:

- `error.py`
- `error_classify.py`
- `error_classify_op.py`

Rule summary:

- `error.py`: move folders containing `bug_report/`
- `error_classify.py`: classify non-accuracy errors by checking `err.log` does not contain `Not equal to tolerance`
- `error_classify_op.py`: classify operator-related errors by checking `err.log` contains one of:
  - `Unsupported operator`
  - `unknown intrinsic Op`
  - `NOT_IMPLEMENTED`

## 4. Training/Test Data Sampling Parameters

### 4.1 Training set construction (`merge_info_train.py`)

`nnsmith_ort_v1.16.0/nnsmith/merge_info_train.py`:

- normal: 1000
- error accuracy: 0
- error other: 1000
- output: `train_1000_1000.json`, `train_time_1000_1000.json`

`nnsmith_ort_v1.17.0/nnsmith/merge_info_train.py`:

- normal: 1000
- error accuracy: 0
- error other: 1000
- output: `train_1000_1000.json`, `train_time_1000_1000.json`

`nnsmith_tvm_v0.10.0/nnsmith/merge_info_train.py`:

- normal: 3000 (random sample)
- error accuracy: 0
- error other: 3000 (random sample)
- output: `train_3000_3000.json`, `train_time_3000_3000.json`

`nnsmith_tvm_v0.11.1/nnsmith/merge_info_train.py`:

- normal: 3000 (random sample)
- error accuracy: 0
- error other: 3000 (random sample)
- output: `train_3000_3000.json` (time json generation is commented in script)

### 4.2 Test set construction (`merge_info.py`)

`nnsmith_ort_v1.16.0/nnsmith/merge_info.py`:

- normal: 100000 (random sample)
- error accuracy: 0 used
- error other: 9 (random sample)
- output folder: `ort-v1.16.0-testdata_0_9`

`nnsmith_ort_v1.17.0/nnsmith/merge_info.py`:

- normal: 90000 (random sample)
- error accuracy: 0
- error other: 7 (random sample)
- output folder: `ort-v1.17.0-testdata_0_7`

`nnsmith_tvm_v0.10.0/nnsmith/merge_info.py`:

- normal: 90000 (random sample)
- error accuracy: 0
- error other: 11 (random sample)
- output folder: `tvm-v0.10.0-testdata_0_11`

`nnsmith_tvm_v0.11.1/nnsmith/merge_info.py`:

- normal: 90000 (random sample)
- error accuracy: 0
- error other: 10 (random sample)
- output folder: `tvm-v0.11.1-testdata_0_11`

## 5. MeDAC (MPNN + Attention) Hyperparameters

Source notebook:

- `nnsmith_ort_v1.17.0/nnsmith/MPNN_edge_message_TCP.ipynb`

Key values in `TrainArgs`:

- `batch_size = 16`
- `num_workers = 16`
- `dataset_type = classification`
- `num_tasks = 1`
- `seed = 0`
- `hidden_size = 300`
- `depth = 3`
- `dropout = 0.3`
- `ffn_num_layers = 3`
- `ffn_hidden_size = 200`
- `init_lr = 1e-4`
- `max_lr = 1e-3`
- `final_lr = 1e-4`
- `warmup_epochs = 2.0`
- `epochs = 100`
- `aggregation = norm`
- `aggregation_norm = 200`

Model structure summary:

- Message passing network with both node-level and edge-level attention
- Graph readout + encoded graph-level features concatenation
- FFN + sigmoid for binary classification

Loss and optimization:

- loss: binary cross entropy (`F.binary_cross_entropy`)
- optimizer: Adam
- scheduler: NoamLR (warmup + exponential decay)

Data split in notebook:

- train/val/test = `0.8 / 0.1 / 0.1`

## 6. LET Baseline Parameters

Source notebook:

- `nnsmith_ort_v1.17.0/nnsmith/LET.ipynb`

Observed design:

- input vector dimension: `74`
- MLP with BatchNorm + Dropout
- output dimension: `1` with sigmoid
- evaluation threshold commonly set to `0.6`
- DataLoader batch size in shown cells: `64`

## 7. GCN Baseline Parameters

Source notebook:

- `nnsmith_ort_v1.17.0/nnsmith/gcn_new.ipynb`

Observed design:

- 3-layer `GCNConv`
- global mean pooling
- FC classifier head to binary sigmoid output
- dropout around `0.2` in shown layers
- NoamLR scheduler class also present in notebook

## 8. Speedup Evaluation Parameters

Source notebook:

- `nnsmith_ort_v1.17.0/nnsmith/MPNN_edge_message_TCP.ipynb`

Method:

1. Obtain predicted ranking for test cases.
2. Simulate execution and accumulate `texec`.
3. Record bug-detection cumulative time list for each method.
4. Compare with random ordering baseline (`RO`).
5. Compute speedup list for LET/GCN/MEPA and ablation.

Cross-scenario aggregation in notebook:

- load speedup pkl files from four settings
- build:
  - `cross_version`
  - `cross_compiler`
- compute and visualize:
  - mean speedup
  - median speedup
  - distributions (bar/violin/histogram)

## 9. Artifacts to Keep

For reproducibility and auditability, keep:

- generated/processed model folders (`normal_models*`, `error_*`)
- merged datasets (`train_*.json`, `*-testdata_*/demo_test_*.json`)
- model checkpoints (`*.pth`)
- evaluation artifacts (`*.pkl`)
- final figures exported from notebooks
