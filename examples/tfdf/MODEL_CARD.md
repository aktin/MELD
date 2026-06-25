# Model Card: Stationary Admission — TF Decision Forests

**Contract:** `stationary-admission-test-tfdf` v0.1.0
**Framework:** TensorFlow Decision Forests (TFDF)
**Runtime Image:** `ghcr.io/simhue/meld/runtime_examples/tfdf:0.1.0`

## Purpose

Binary classifier that predicts whether an emergency department patient will be admitted to inpatient care (`admitted: 0 or 1`).

This is a **testing example** for the MELD framework, demonstrating how to package a gradient-boosted tree model for managed inference.

## Model Architecture

TF-DF `GradientBoostedTreesModel` — trained directly on a Pandas dataframe via `pd_dataframe_to_tf_dataset`. No manual feature preprocessing required; TFDF handles mixed types natively. Saved as a TensorFlow SavedModel.

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
cd examples/tfdf/train
python train.py   # reads ./input.csv and ../resources/contract-training.yaml
                  # writes SavedModel artifact to ../artifact/
```

## Inference

Runs inside a Docker container managed by MELD. Reads from `/input/input.csv` + `/input/contract.yaml`, writes to `/output/output.csv`. Uses `serving_default` signature from the SavedModel.

To run via MELD, reference `examples/tfdf/resources/contract.yaml` as the contract.
