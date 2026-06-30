# Model Card: Stationary Admission — scikit-learn Random Forest

**Contract:** `stationary-admission-test-sklearn` v0.1.0
**Framework:** scikit-learn
**Runtime Image:** `ghcr.io/simhue/meld/runtime_examples/sklearn:0.1.0`

## Purpose

Binary classifier that predicts whether an emergency department patient will be admitted to inpatient care (`admitted: 0 or 1`).

This is a **testing example** for the MELD framework, demonstrating how to package a scikit-learn model for managed inference.

## Model Architecture

scikit-learn `Pipeline` consisting of:
- `ColumnTransformer`: `StandardScaler` for numeric features, `OrdinalEncoder` for categorical features
- `RandomForestClassifier` with 100 estimators
- Serialized to disk with `joblib`

## Input Features

| Feature | Type | Description |
|---|---|---|
| `cedis_code` | string | CEDIS triage category code |
| `age` | string | Patient age group |
| `admission_time` | datetime | Timestamp of admission (converted to Unix seconds) |
| `triage_score` | integer | Triage severity score |

**Temporal scope:** relative, `-P1M` from anchor date

## Output

| Field | Type | Description |
|---|---|---|
| `admitted` | integer | 1 = admitted, 0 = not admitted |

Output format: CSV

## Training

```bash
cd examples/sklearn/train
python train.py   # reads ./input.csv and ../resources/contract-training.yaml
                  # writes artifact to ../artifact/model.joblib
```

Copy `input.csv` from another example's `train/` directory (same synthesized dataset).

## Inference

Runs inside a Docker container managed by MELD. Reads from `/input/input.csv` + `/input/contract.yaml`, writes to `/output/output.csv`.

To run via MELD, reference `examples/sklearn/resources/contract.yaml` as the contract.
