# Model Card: Stationary Admission — Neural Network

**Contract:** `stationary-admission-test-nn` v0.2.0
**Framework:** TensorFlow (Keras)
**Runtime Image:** `ghcr.io/simhue/meld/runtime_examples/nn:0.2.0`

## Purpose

Binary classifier that predicts whether an emergency department patient will be admitted to inpatient care (`admitted: 0 or 1`).

This is a **testing example** for the MELD framework, demonstrating how to package a neural network model for managed inference.

## Model Architecture

Keras functional API model:
- Per-feature preprocessing: `Normalization` for numeric, `StringLookup` + one-hot for categorical
- Dense(64, relu) → Dropout(0.2) → Dense(32, relu)
- Output: sigmoid (binary) or softmax (multi-class)
- Optimizer: Adam, trained for 20 epochs

## Input Features

| Feature | Type | Description |
|---|---|---|
| `cedis_code` | string | CEDIS triage category code |
| `age` | string | Patient age group |
| `admission_time` | datetime | Timestamp of admission |
| `triage_score` | integer | Triage severity score |

**Temporal scope:** relative, `-P1M` from anchor date

## Output

| Field | Type | Description |
|---|---|---|
| `admitted` | integer | 1 = admitted, 0 = not admitted |

Output format: CSV

## Training

```bash
cd examples/nn/train
python train.py   # reads ./input.csv and ../resources/contract-training.yaml
                  # writes artifact to ../artifact/model.keras
```

## Inference

Runs inside a Docker container managed by MELD. Reads from `/input/input.csv` + `/input/contract.yaml`, writes to `/output/output.csv`.

To run via MELD, reference `examples/nn/resources/contract.yaml` as the contract.
